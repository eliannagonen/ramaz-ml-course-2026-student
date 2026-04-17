from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

# Part 1 function names that must not use PyTorch.
_MANUAL_FN_NAMES: list[str] = [
    "manual_gradient_quadratic",
    "manual_gradient_composite",
    "manual_partial_gradients",
]


def _fn_uses_torch(fn: object) -> bool:
    """Return True if the function's AST contains any 'torch.*' attribute access."""
    try:
        source = textwrap.dedent(inspect.getsource(fn))  # type: ignore[arg-type]
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "torch"
        ):
            return True
    return False


@pytest.fixture(autouse=True)
def set_seed() -> None:
    """Set deterministic seeds for reproducible PyTorch results."""
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True


@pytest.fixture(autouse=True)
def check_no_torch_in_manual(request: pytest.FixtureRequest) -> None:
    """Fail if any Part 1 (manual) function uses torch internally."""
    if "manual" not in request.node.keywords:
        return
    import derivatives as d

    for fn_name in _MANUAL_FN_NAMES:
        fn = getattr(d, fn_name, None)
        if fn is not None and _fn_uses_torch(fn):
            pytest.fail(
                f"Part 1 function '{fn_name}' uses 'torch'. "
                "Part 1 must be implemented using only Python arithmetic — "
                "no torch, no math library. Use the derivative rules directly."
            )
