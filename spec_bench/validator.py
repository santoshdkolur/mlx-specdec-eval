"""
Pre-Flight Validator Module for mlx-specdec-eval.
Executes fail-fast environment checks, tokenizer compatibility verification, and Apple Silicon RAM footprint estimation.
"""

import sys
import os
import platform
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import psutil

try:
    import mlx.core as mx
    import mlx.core.metal as metal
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    metal = None

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from huggingface_hub import HfApi
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False


def validate_system_environment() -> Dict[str, Any]:
    """
    Checks system environment requirements:
    - Python version >= 3.10
    - Active virtual environment (.venv)
    - macOS / Darwin platform
    - Apple Silicon Metal backend availability
    """
    results = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_ok": sys.version_info >= (3, 10),
        "in_venv": sys.prefix != sys.base_prefix,
        "is_darwin": sys.platform == "darwin",
        "architecture": platform.machine(),
        "has_mlx": HAS_MLX,
        "metal_available": metal.is_available() if HAS_MLX and hasattr(metal, "is_available") else False,
        "all_passed": False
    }

    results["all_passed"] = (
        results["python_ok"] and
        results["is_darwin"] and
        results["has_mlx"] and
        results["metal_available"]
    )
    return results


def resolve_model_size_gb(model_path_or_id: str) -> float:
    """
    Estimates the model weight size in GB.
    Supports local directory paths, Hugging Face Hub metadata API,
    and dynamic parameter/quantization regex extraction.
    """
    import re

    # 1. Local directory check
    path = Path(model_path_or_id)
    if path.exists() and path.is_dir():
        total_bytes = sum(
            f.stat().st_size
            for f in path.rglob("*")
            if f.is_file() and f.suffix in ['.safetensors', '.npz', '.bin', '.pt']
        )
        if total_bytes > 0:
            return round(total_bytes / (1024 ** 3), 2)

    # 2. Attempt HF Hub metadata lookup if HF Hub API is available
    if HAS_HF_HUB:
        try:
            api = HfApi()
            model_info = api.model_info(model_path_or_id)
            total_bytes = 0
            if model_info.siblings:
                for sibling in model_info.siblings:
                    size = getattr(sibling, "size", None)
                    if not size and hasattr(sibling, "lfs") and isinstance(sibling.lfs, dict):
                        size = sibling.lfs.get("size")
                    if size:
                        total_bytes += size
            if total_bytes > 0:
                return round(total_bytes / (1024 ** 3), 2)
        except Exception:
            pass

    # 3. Dynamic parameter & quantization extraction (e.g. 0.8b, 2b, 9b, 14b, 4bit, 8bit)
    lower_id = model_path_or_id.lower()
    param_match = re.search(r'(\d+(?:\.\d+)?)\s*b', lower_id)
    if param_match:
        params_b = float(param_match.group(1))
        if "8bit" in lower_id or "fp8" in lower_id or "int8" in lower_id:
            bytes_per_param = 1.05
        elif "16bit" in lower_id or "bf16" in lower_id or "fp16" in lower_id:
            bytes_per_param = 2.05
        else:
            # Default 4-bit quantized MLX weights
            bytes_per_param = 0.58
        return round(params_b * bytes_per_param, 2)

    return 2.5




def validate_tokenizer_compatibility(target_path: str, draft_path: str) -> Tuple[bool, List[str]]:
    """
    Validates tokenizer vocabulary equality, special token IDs, and sample encoding identity.
    Returns (is_compatible, list_of_error_messages).
    """
    errors = []

    if not HAS_TRANSFORMERS:
        return False, ["❌ transformers library is not installed."]

    try:
        tok_target = AutoTokenizer.from_pretrained(target_path)
    except Exception as e:
        return False, [f"❌ Failed to load target tokenizer '{target_path}': {e}"]

    try:
        tok_draft = AutoTokenizer.from_pretrained(draft_path)
    except Exception as e:
        return False, [f"❌ Failed to load draft tokenizer '{draft_path}': {e}"]

    # 1. Vocab Size Check
    if len(tok_target) != len(tok_draft):
        errors.append(
            f"Vocab Mismatch: Target tokenizer size ({len(tok_target)}) != Draft tokenizer size ({len(tok_draft)})."
        )

    # 2. EOS Token Check
    if tok_target.eos_token_id != tok_draft.eos_token_id:
        errors.append(
            f"EOS Token Mismatch: Target EOS ({tok_target.eos_token_id}) != Draft EOS ({tok_draft.eos_token_id})."
        )

    # 3. BOS Token Check
    if tok_target.bos_token_id != tok_draft.bos_token_id:
        errors.append(
            f"BOS Token Mismatch: Target BOS ({tok_target.bos_token_id}) != Draft BOS ({tok_draft.bos_token_id})."
        )

    # 4. Special Control Tokens Identity Check
    common_control_tokens = [
        "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>",
        "<|start_header_id|>", "<|end_header_id|>", "<s>", "</s>", "<pad>"
    ]
    for token_str in common_control_tokens:
        id_t = tok_target.convert_tokens_to_ids(token_str)
        id_d = tok_draft.convert_tokens_to_ids(token_str)
        # If token exists in target tokenizer, verify draft matches it
        if id_t is not None and id_t != tok_target.unk_token_id:
            if id_t != id_d:
                errors.append(
                    f"Control Token Mismatch for '{token_str}': Target ID ({id_t}) != Draft ID ({id_d})."
                )

    # 5. Chat Template Formatting Sweep
    if hasattr(tok_target, "apply_chat_template") and hasattr(tok_draft, "apply_chat_template"):
        try:
            test_msg = [{"role": "user", "content": "spec_dec_check"}]
            formatted_t = tok_target.apply_chat_template(test_msg, tokenize=False)
            formatted_d = tok_draft.apply_chat_template(test_msg, tokenize=False)
            if tok_target.encode(formatted_t) != tok_draft.encode(formatted_d):
                errors.append(
                    "Chat Template Formatting Mismatch: Target and Draft produce different token sequences for chat templates."
                )
        except Exception:
            pass

    # 6. Sample Encoding Identity Sweep
    sample_texts = [
        "def benchmark_test(): return 'speculative_decoding_test_123'",
        "The quick brown fox jumps over the lazy dog.",
        "System prompt test: 1234567890 !@#$%^&*()"
    ]

    for sample in sample_texts:
        enc_t = tok_target.encode(sample)
        enc_d = tok_draft.encode(sample)
        if enc_t != enc_d:
            errors.append(
                f"Encoding Mismatch for sample text '{sample[:30]}...': Target IDs != Draft IDs."
            )
            break

    return len(errors) == 0, errors



def validate_memory_footprint(
    target_path: str,
    draft_path: str,
    safety_buffer_gb: float = 1.0
) -> Dict[str, Any]:
    """
    Calculates estimated VRAM/RAM footprint and compares against Apple Silicon Unified Memory limits.
    On macOS Apple Silicon, MLX can allocate up to ~85% of total unified RAM.
    """
    ram_info = psutil.virtual_memory()
    available_ram_gb = ram_info.available / (1024 ** 3)
    total_ram_gb = ram_info.total / (1024 ** 3)

    target_size_gb = resolve_model_size_gb(target_path)
    draft_size_gb = resolve_model_size_gb(draft_path)
    estimated_kv_cache_gb = 0.5

    total_model_footprint_gb = target_size_gb + draft_size_gb + estimated_kv_cache_gb
    max_allocatable_ram_gb = total_ram_gb * 0.85
    is_safe = max_allocatable_ram_gb >= (total_model_footprint_gb + safety_buffer_gb)

    return {
        "available_ram_gb": round(available_ram_gb, 2),
        "total_ram_gb": round(total_ram_gb, 2),
        "target_size_gb": round(target_size_gb, 2),
        "draft_size_gb": round(draft_size_gb, 2),
        "estimated_kv_cache_gb": round(estimated_kv_cache_gb, 2),
        "total_required_gb": round(total_model_footprint_gb, 2),
        "is_safe": is_safe,
        "is_tight": available_ram_gb < total_model_footprint_gb
    }

