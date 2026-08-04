"""
MLX Speculative Decoding Engine.
Handles loading target and draft models, executing baseline and speculative generation loops,
and recording exact per-step metrics (prefill latency, decode latency, draft acceptance rates, peak Metal VRAM).
"""

import time
from typing import Dict, Any, List, Optional, Tuple
import mlx.core as mx

try:
    import mlx.core.metal as metal
except ImportError:
    metal = None

try:
    from mlx_lm import load, generate, stream_generate
    try:
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        make_sampler = None
    HAS_MLX_LM = True
except ImportError:
    make_sampler = None
    stream_generate = None
    HAS_MLX_LM = False


def reset_peak_mem():
    """Resets Metal peak memory across MLX framework versions."""
    if hasattr(mx, "reset_peak_memory"):
        try:
            mx.reset_peak_memory()
            return
        except Exception:
            pass
    if metal and hasattr(metal, "reset_peak_memory"):
        try:
            metal.reset_peak_memory()
        except Exception:
            pass


def get_peak_mem_mb() -> float:
    """Returns peak Metal memory usage in MB."""
    bytes_val = 0
    if hasattr(mx, "get_peak_memory"):
        try:
            bytes_val = mx.get_peak_memory()
        except Exception:
            pass
    if bytes_val == 0 and metal and hasattr(metal, "get_peak_memory"):
        try:
            bytes_val = metal.get_peak_memory()
        except Exception:
            pass
    return bytes_val / (1024 * 1024)


class MLXSpeculativeEngine:
    """
    Speculative decoding runner using MLX framework.
    """

    def __init__(self, target_model_path: str, draft_model_path: str):
        self.target_model_path = target_model_path
        self.draft_model_path = draft_model_path
        self.target_model = None
        self.tokenizer = None
        self.draft_model = None
        self.is_loaded = False

    def load_models(self):
        """Loads target and draft models into MLX Metal Unified Memory."""
        if not HAS_MLX_LM:
            raise RuntimeError("❌ mlx-lm library is not installed.")

        print(f"📦 Loading Target Model: {self.target_model_path}...")
        self.target_model, self.tokenizer = load(self.target_model_path)

        print(f"📦 Loading Draft Model:  {self.draft_model_path}...")
        self.draft_model, _ = load(self.draft_model_path)

        self.is_loaded = True
        print("✅ Models loaded successfully into Metal Unified Memory.")

    def run_baseline(
        self,
        prompt: str,
        max_tokens: int = 128,
        temp: float = 0.0
    ) -> Dict[str, Any]:
        """
        Executes standard non-speculative autoregressive generation on the target model.
        """
        if not self.is_loaded:
            self.load_models()

        # Automatically format prompt using model's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        reset_peak_mem()


        gen_kwargs = {
            "prompt": prompt,
            "max_tokens": max_tokens
        }

        if temp > 0.0 and make_sampler is not None:
            gen_kwargs["sampler"] = make_sampler(temp)

        start_time = time.perf_counter()
        first_token_time = None
        output_tokens = 0
        response = ""

        if HAS_MLX_LM and stream_generate is not None:
            for response_chunk in stream_generate(
                self.target_model,
                self.tokenizer,
                **gen_kwargs
            ):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                output_tokens += 1
                response += response_chunk.text
        else:
            response = generate(
                self.target_model,
                self.tokenizer,
                **gen_kwargs
            )
            output_tokens = max(1, len(self.tokenizer.encode(response)))

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        prefill_time = (first_token_time - start_time) if first_token_time else 0.0
        decode_time = (end_time - first_token_time) if first_token_time and end_time > first_token_time else elapsed_time

        # Generation Decode TPS (standard LLM benchmark metric)
        tps = output_tokens / decode_time if decode_time > 0 else 0.0
        total_tps = output_tokens / elapsed_time if elapsed_time > 0 else 0.0
        peak_memory_mb = get_peak_mem_mb()

        return {
            "mode": "baseline",
            "prompt": prompt,
            "text": response,
            "output_tokens": output_tokens,
            "elapsed_time_s": elapsed_time,
            "prefill_time_s": prefill_time,
            "decode_time_s": decode_time,
            "tps": tps,
            "total_tps": total_tps,
            "peak_memory_mb": peak_memory_mb
        }


    def run_speculative(
        self,
        prompt: str,
        num_draft_tokens: int = 5,
        max_tokens: int = 128,
        temp: float = 0.0
    ) -> Dict[str, Any]:
        """
        Executes speculative decoding generation using target and draft models.
        Tracks total draft tokens proposed, accepted tokens, acceptance rate (alpha),
        prefill vs decode latencies, and peak Metal memory.
        """
        if not self.is_loaded:
            self.load_models()

        # Automatically format prompt using model's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        reset_peak_mem()

        gen_kwargs = {
            "prompt": prompt,
            "draft_model": self.draft_model,

            "num_draft_tokens": num_draft_tokens,
            "max_tokens": max_tokens
        }

        if temp > 0.0 and make_sampler is not None:
            gen_kwargs["sampler"] = make_sampler(temp)

        accepted_draft_tokens = 0
        total_draft_tokens_proposed = 0
        output_tokens = 0
        response_text = ""
        in_draft_step = False

        start_time = time.perf_counter()
        first_token_time = None

        if HAS_MLX_LM and stream_generate is not None:
            for response_chunk in stream_generate(
                self.target_model,
                self.tokenizer,
                **gen_kwargs
            ):
                if first_token_time is None:
                    first_token_time = time.perf_counter()

                if not in_draft_step:
                    remaining = max_tokens - output_tokens
                    proposed_this_step = min(remaining, num_draft_tokens)
                    total_draft_tokens_proposed += proposed_this_step
                    in_draft_step = True

                if getattr(response_chunk, "from_draft", False):
                    accepted_draft_tokens += 1
                else:
                    in_draft_step = False

                output_tokens += 1
                response_text += response_chunk.text

        else:
            response_text = generate(
                self.target_model,
                self.tokenizer,
                **gen_kwargs
            )
            # Same: generate() returns completion only, not prompt+completion.
            output_tokens = max(1, len(self.tokenizer.encode(response_text)))

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        prefill_time = (first_token_time - start_time) if first_token_time else 0.0
        decode_time = (end_time - first_token_time) if first_token_time and end_time > first_token_time else elapsed_time

        # Generation Decode TPS (standard LLM benchmark metric)
        tps = output_tokens / decode_time if decode_time > 0 else 0.0
        total_tps = output_tokens / elapsed_time if elapsed_time > 0 else 0.0
        peak_memory_mb = get_peak_mem_mb()

        alpha = accepted_draft_tokens / total_draft_tokens_proposed if total_draft_tokens_proposed > 0 else 0.0

        return {
            "mode": "speculative",
            "prompt": prompt,
            "text": response_text,
            "num_draft_tokens": num_draft_tokens,
            "output_tokens": output_tokens,
            "total_draft_proposed": total_draft_tokens_proposed,
            "accepted_draft_tokens": accepted_draft_tokens,
            "alpha": alpha,
            "elapsed_time_s": elapsed_time,
            "prefill_time_s": prefill_time,
            "decode_time_s": decode_time,
            "tps": tps,
            "total_tps": total_tps,
            "peak_memory_mb": peak_memory_mb
        }

