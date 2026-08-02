"""
Unit tests for spec_bench.validator module.
"""

import sys
from spec_bench.validator import (
    validate_system_environment,
    resolve_model_size_gb,
    validate_memory_footprint
)


def test_validate_system_environment():
    res = validate_system_environment()
    assert "python_version" in res
    assert "python_ok" in res
    assert res["python_ok"] is True
    assert "is_darwin" in res
    assert "has_mlx" in res


def test_resolve_model_size_gb_nonexistent():
    # Fallback default size for nonexistent model path
    size = resolve_model_size_gb("non_existent_model_xyz_123")
    assert size > 0.0


def test_validate_memory_footprint():
    mem_info = validate_memory_footprint("non_existent_target", "non_existent_draft", safety_buffer_gb=2.0)
    assert "available_ram_gb" in mem_info
    assert "total_required_gb" in mem_info
    assert "is_safe" in mem_info
    assert isinstance(mem_info["is_safe"], bool)
