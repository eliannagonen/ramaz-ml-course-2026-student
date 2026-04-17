"""Score calculator for HW02.

Usage:
    uv run python score.py          # full score
    uv run python score.py manual   # Part 1 only
    uv run python score.py autograd # Part 2 only
"""

from __future__ import annotations

import subprocess
import sys

# Points per test class — all-or-nothing within each class.
SECTIONS: list[tuple[str, str, int]] = [
    ("TestManualGradientQuadratic", "manual", 10),
    ("TestManualGradientComposite", "manual", 10),
    ("TestManualPartialGradients", "manual", 10),
    ("TestAutogradGradientQuadratic", "autograd", 10),
    ("TestAutogradGradientComposite", "autograd", 10),
    ("TestAutogradPartialGradients", "autograd", 10),
]

TOTAL_POINTS = sum(pts for _, _, pts in SECTIONS)

# Label shown to the student (snake_case version of the class)
_LABELS: dict[str, str] = {
    "TestManualGradientQuadratic": "manual_gradient_quadratic",
    "TestManualGradientComposite": "manual_gradient_composite",
    "TestManualPartialGradients": "manual_partial_gradients",
    "TestAutogradGradientQuadratic": "autograd_gradient_quadratic",
    "TestAutogradGradientComposite": "autograd_gradient_composite",
    "TestAutogradPartialGradients": "autograd_partial_gradients",
}


def score_by_class() -> int:
    """Run pytest per test class and accumulate points (all-or-nothing per class)."""
    earned = 0
    print(f"\n{'Function':<35} {'Status':<10} {'Points':>6}")
    print("-" * 55)

    for class_name, _marker, pts in SECTIONS:
        cmd = [sys.executable, "-m", "pytest", "--tb=no", "-q", "-k", class_name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout + result.stderr
        passed = output.count(" passed")
        failed = output.count(" failed") + output.count(" error")
        total = passed + failed

        if total > 0 and failed == 0:
            status = "PASS"
            earned += pts
        elif total == 0:
            status = "NO TESTS"
        else:
            status = f"FAIL ({failed}/{total})"

        label = _LABELS[class_name]
        pts_str = f"{pts if (total > 0 and failed == 0) else 0}/{pts}"
        print(f"{label:<35} {status:<10} {pts_str:>6}")

    print("-" * 55)
    print(f"{'TOTAL':<35} {'':10} {earned}/{TOTAL_POINTS}")
    return earned


if __name__ == "__main__":
    marker_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if marker_arg == "manual":
        print("=== Part 1: Manual Derivatives ===")
        cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short", "-m", "manual"]
        subprocess.run(cmd)
    elif marker_arg == "autograd":
        print("=== Part 2: Autograd ===")
        cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short", "-m", "autograd"]
        subprocess.run(cmd)
    else:
        score_by_class()
