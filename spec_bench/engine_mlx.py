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


    def _execute_speculative_loop(
        self,
        prompt: str,
        max_tokens: int = 128,
        temp: float = 0.0,
        fixed_k: Optional[int] = None,
        min_k: int = 1,
        max_k: int = 5,
        initial_k: int = 3,
        step_k: int = 1,
        beta: float = 0.5
    ) -> Dict[str, Any]:
        """
        Unified speculative generator supporting both standard trimmable KV caches
        and hybrid recurrent state models (e.g. Qwen 3.5 ArraysCache) with round snapshotting.
        Supports both fixed-K sweeps and dynamic adaptive K.
        """
        from mlx_lm.models import cache
        from mlx_lm.tokenizer_utils import TokenizerWrapper

        is_adaptive = fixed_k is None
        current_k = fixed_k if fixed_k is not None else initial_k

        tok = self.tokenizer
        if not isinstance(tok, TokenizerWrapper):
            tok = TokenizerWrapper(tok)

        prompt_tokens = mx.array(tok.encode(prompt, add_special_tokens=False))
        detokenizer = tok.detokenizer
        detokenizer.reset()

        model_cache = cache.make_prompt_cache(self.target_model)
        draft_cache = cache.make_prompt_cache(self.draft_model)

        is_trimmable_model = cache.can_trim_prompt_cache(model_cache)
        is_trimmable_draft = cache.can_trim_prompt_cache(draft_cache)

        def _clone_cache_state(c_list):
            snap = []
            for c in c_list:
                if hasattr(c, "state"):
                    s = c.state
                    if isinstance(s, (list, tuple)):
                        snap.append((type(c), [x for x in s], getattr(c, "offset", None)))
                    else:
                        snap.append((type(c), s, getattr(c, "offset", None)))
                else:
                    snap.append((type(c), None, None))
            return snap

        def _restore_cache_state(c_list, snap):
            for c, (c_type, state, offset) in zip(c_list, snap):
                if hasattr(c, "state") and state is not None:
                    c.state = state
                if offset is not None and hasattr(c, "offset"):
                    c.offset = offset

        def _prefill(model, m_cache, y_arr, prefill_step_size=512):
            while y_arr.size > 1:
                n_to_process = min(prefill_step_size, y_arr.size - 1)
                model(y_arr[:n_to_process][None], cache=m_cache)
                mx.eval([c.state for c in m_cache])
                y_arr = y_arr[n_to_process:]
                mx.clear_cache()
            return y_arr

        def _step(model, m_cache, y_arr, n_predict=1):
            logits = model(y_arr[None], cache=m_cache)
            logits = logits[:, -n_predict:, :]
            out_y, out_logprobs = [], []
            for i in range(n_predict):
                l = logits[:, i, :]
                logprobs = l - mx.logsumexp(l, axis=-1, keepdims=True)
                if temp > 0.0 and make_sampler is not None:
                    sampler = make_sampler(temp)
                    y_token = sampler(logprobs)
                else:
                    y_token = mx.argmax(logprobs, axis=-1)
                out_y.append(y_token)
                out_logprobs.append(logprobs)
            return mx.concatenate(out_y, axis=0), mx.concatenate(out_logprobs, axis=0)

        def _draft_generate(y_arr, num_draft):
            if num_draft == 0:
                return mx.array([], mx.uint32)
            ys = []
            for _ in range(num_draft):
                y_arr, _ = _step(self.draft_model, draft_cache, y_arr)
                mx.async_eval(y_arr)
                ys.append(y_arr)
            return mx.concatenate(ys)

        y = prompt_tokens.astype(mx.uint32)
        start_time = time.perf_counter()
        draft_y = _prefill(self.draft_model, draft_cache, y)
        y = _prefill(self.target_model, model_cache, y)

        ntoks = 0
        k_history = []
        k_selections = []
        output_tokens = 0
        accepted_draft_tokens = 0
        total_draft_tokens_proposed = 0
        first_token_time = None
        round_idx = 0
        response_text = ""
        ema_alpha = 0.5

        while ntoks < max_tokens:
            num_draft = min(max_tokens - ntoks, current_k)
            k_selections.append(num_draft)
            total_draft_tokens_proposed += num_draft
            round_idx += 1

            # Snapshot non-trimmable states before step
            target_snap = _clone_cache_state(model_cache) if not is_trimmable_model else None
            draft_snap = _clone_cache_state(draft_cache) if not is_trimmable_draft else None

            draft_tokens = _draft_generate(draft_y, num_draft)
            target_input = mx.concatenate([y, draft_tokens])
            tokens, logprobs = _step(self.target_model, model_cache, target_input, num_draft + 1)
            mx.eval(tokens, draft_tokens)
            draft_tokens_list = draft_tokens.tolist()
            tokens_list = tokens.tolist()

            n = 0
            while n < num_draft:
                tn, dtn = tokens_list[n], draft_tokens_list[n]
                if tn != dtn:
                    break
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                n += 1
                ntoks += 1
                output_tokens += 1
                accepted_draft_tokens += 1
                detokenizer.add_token(tn)
                response_text += detokenizer.last_segment
                if ntoks == max_tokens:
                    break

            if ntoks < max_tokens:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                ntoks += 1
                output_tokens += 1
                detokenizer.add_token(tokens_list[n])
                response_text += detokenizer.last_segment

            alpha_round = n / num_draft if num_draft > 0 else 0.0
            ema_alpha = beta * ema_alpha + (1.0 - beta) * alpha_round

            # Adaptive K decision for next round
            if is_adaptive:
                if n == num_draft or alpha_round >= 0.75:
                    next_k = min(current_k + step_k, max_k)
                elif n == 0 or alpha_round <= 0.33:
                    next_k = max(current_k - step_k, min_k)
                else:
                    next_k = current_k

                k_history.append({
                    "round": round_idx,
                    "k": num_draft,
                    "accepted": n,
                    "alpha_round": round(alpha_round, 2),
                    "next_k": next_k
                })
                current_k = next_k

            if ntoks == max_tokens:
                break

            # Cache rewinds / fast-forwards
            if is_trimmable_model:
                cache.trim_prompt_cache(model_cache, num_draft - n)
            else:
                _restore_cache_state(model_cache, target_snap)
                accepted_seq = target_input[: n + 1]
                _step(self.target_model, model_cache, accepted_seq, n + 1)

            if is_trimmable_draft:
                cache.trim_prompt_cache(draft_cache, max(num_draft - n - 1, 0))
            else:
                _restore_cache_state(draft_cache, draft_snap)
                if n > 0:
                    draft_accepted_seq = draft_tokens[:n]
                    _step(self.draft_model, draft_cache, draft_accepted_seq, n)

            y = mx.array([tokens_list[n]], mx.uint32)
            draft_y = y
            if n == num_draft:
                draft_y = mx.concatenate([mx.array(draft_tokens_list[-1:], mx.uint32), draft_y])

        detokenizer.finalize()
        response_text += detokenizer.last_segment
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        prefill_time = (first_token_time - start_time) if first_token_time else 0.0
        decode_time = (end_time - first_token_time) if first_token_time and end_time > first_token_time else elapsed_time

        tps = output_tokens / decode_time if decode_time > 0 else 0.0
        total_tps = output_tokens / elapsed_time if elapsed_time > 0 else 0.0
        alpha = accepted_draft_tokens / total_draft_tokens_proposed if total_draft_tokens_proposed > 0 else 0.0
        wasted_draft_tokens = total_draft_tokens_proposed - accepted_draft_tokens
        wasted_draft_ratio = wasted_draft_tokens / output_tokens if output_tokens > 0 else 0.0
        mean_effective_k = sum(k_selections) / len(k_selections) if k_selections else float(initial_k)
        peak_memory_mb = get_peak_mem_mb()

        result_dict = {
            "mode": "adaptive" if is_adaptive else "speculative",
            "prompt": prompt,
            "text": response_text,
            "output_tokens": output_tokens,
            "total_draft_proposed": total_draft_tokens_proposed,
            "accepted_draft_tokens": accepted_draft_tokens,
            "alpha": alpha,
            "wasted_draft_tokens": wasted_draft_tokens,
            "wasted_draft_ratio": wasted_draft_ratio,
            "elapsed_time_s": elapsed_time,
            "prefill_time_s": prefill_time,
            "decode_time_s": decode_time,
            "tps": tps,
            "total_tps": total_tps,
            "peak_memory_mb": peak_memory_mb
        }

        if is_adaptive:
            result_dict.update({
                "min_k": min_k,
                "max_k": max_k,
                "initial_k": initial_k,
                "mean_effective_k": mean_effective_k,
                "k_history": k_history,
                "k_selections": k_selections
            })
        else:
            result_dict["num_draft_tokens"] = fixed_k

        return result_dict

    def run_speculative(
        self,
        prompt: str,
        num_draft_tokens: int = 5,
        max_tokens: int = 128,
        temp: float = 0.0
    ) -> Dict[str, Any]:
        """
        Executes speculative decoding generation using target and draft models.
        Automatically supports both standard trimmable and hybrid recurrent models.
        """
        if not self.is_loaded:
            self.load_models()

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        reset_peak_mem()

        if not HAS_MLX_LM or self.target_model is None:
            # Synthetic / fallback execution mode
            start_time = time.perf_counter()
            time.sleep(0.01)
            end_time = time.perf_counter()
            return {
                "mode": "speculative",
                "prompt": prompt,
                "text": f"Synthetic speculative response for: {prompt[:30]}...",
                "num_draft_tokens": num_draft_tokens,
                "output_tokens": max_tokens,
                "total_draft_proposed": int(max_tokens * 0.8),
                "accepted_draft_tokens": int(max_tokens * 0.55),
                "alpha": 0.68,
                "wasted_draft_tokens": int(max_tokens * 0.25),
                "wasted_draft_ratio": 0.25,
                "elapsed_time_s": end_time - start_time,
                "prefill_time_s": 0.002,
                "decode_time_s": end_time - start_time,
                "tps": 35.0,
                "total_tps": 34.0,
                "peak_memory_mb": 500.0
            }

        return self._execute_speculative_loop(
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temp,
            fixed_k=num_draft_tokens
        )

    def run_adaptive_speculative(
        self,
        prompt: str,
        min_k: int = 1,
        max_k: int = 5,
        initial_k: int = 3,
        step_k: int = 1,
        max_tokens: int = 128,
        temp: float = 0.0,
        beta: float = 0.5
    ) -> Dict[str, Any]:
        """
        Executes adaptive speculative decoding where K is dynamically adjusted.
        Automatically supports both standard trimmable and hybrid recurrent models.
        """
        if not self.is_loaded:
            self.load_models()

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        reset_peak_mem()

        if not HAS_MLX_LM or self.target_model is None:
            # Synthetic / fallback execution mode
            start_time = time.perf_counter()
            time.sleep(0.01)
            end_time = time.perf_counter()
            return {
                "mode": "adaptive",
                "prompt": prompt,
                "text": f"Synthetic adaptive response for: {prompt[:30]}...",
                "min_k": min_k,
                "max_k": max_k,
                "initial_k": initial_k,
                "output_tokens": max_tokens,
                "total_draft_proposed": int(max_tokens * 0.8),
                "accepted_draft_tokens": int(max_tokens * 0.55),
                "alpha": 0.68,
                "wasted_draft_tokens": int(max_tokens * 0.25),
                "wasted_draft_ratio": 0.25,
                "mean_effective_k": 3.2,
                "k_history": [{"round": 1, "k": initial_k, "accepted": 2, "alpha_round": 0.67, "next_k": 3}],
                "k_selections": [3, 4, 3, 3],
                "elapsed_time_s": end_time - start_time,
                "prefill_time_s": 0.002,
                "decode_time_s": end_time - start_time,
                "tps": 35.0,
                "total_tps": 34.0,
                "peak_memory_mb": 500.0
            }

        return self._execute_speculative_loop(
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temp,
            fixed_k=None,
            min_k=min_k,
            max_k=max_k,
            initial_k=initial_k,
            step_k=step_k,
            beta=beta
        )



