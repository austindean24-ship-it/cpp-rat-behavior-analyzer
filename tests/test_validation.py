from __future__ import annotations

from validate_demo import run_validation


def test_synthetic_validation_stays_within_reasonable_error(tmp_path) -> None:
    results = run_validation(output_dir=tmp_path)
    assert results["max_abs_seconds_error"] <= 0.75
    assert results["max_abs_percent_error"] <= 20.0
