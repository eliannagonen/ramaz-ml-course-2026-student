"""HW02 — Derivatives and Autograd

Part 1: Manual derivative implementations — translate the derivative rules
directly into Python arithmetic. No torch allowed.

Part 2: Autograd implementations — use torch.autograd to compute the same
gradients automatically.

Run tests:   uv run pytest
Check score: uv run python score.py
"""

from __future__ import annotations

import torch

# ── Part 1: Manual Derivatives ────────────────────────────────────────────────


def manual_gradient_quadratic(a: float, b: float, x_val: float) -> float:
    """Return d/dx of f(x) = a*x**2 + b*x evaluated at x = x_val.

    Compute this by hand using the power rule — no torch, no math library
    beyond basic arithmetic.

    Args:
        a: Coefficient of the x**2 term.
        b: Coefficient of the x term.
        x_val: The point at which to evaluate the derivative.

    Returns:
        The value of f'(x_val).

    Example:
        >>> manual_gradient_quadratic(1.0, 0.0, 3.0)
        6.0
        # f(x) = x**2  =>  f'(x) = 2x  =>  f'(3) = 6

    Hint:
        Apply the power rule to each term separately, then add.
        Recall: d/dx [c * x**n] = c * n * x**(n-1).
    """
    raise NotImplementedError("Implement manual_gradient_quadratic()")


def manual_gradient_composite(x_val: float) -> float:
    """Return d/dx of f(x) = (x**2 + 1)**3 evaluated at x = x_val.

    Compute this using the chain rule — no torch, no math library beyond
    basic arithmetic.

    Args:
        x_val: The point at which to evaluate the derivative.

    Returns:
        The value of f'(x_val).

    Example:
        >>> manual_gradient_composite(1.0)
        24.0
        # f'(1) = 24

    Hint:
        Identify the outer function (something cubed) and inner function
        (x**2 + 1). Chain rule: d/dx outer(inner) = outer'(inner) * inner'.
    """
    raise NotImplementedError("Implement manual_gradient_composite()")


def manual_partial_gradients(x_val: float, y_val: float) -> tuple[float, float]:
    """Return (df/dx, df/dy) of f(x, y) = x**2 + x*y + y**2 at (x_val, y_val).

    Compute both partial derivatives by hand — no torch.

    Args:
        x_val: x-coordinate of the evaluation point.
        y_val: y-coordinate of the evaluation point.

    Returns:
        A tuple (df_dx, df_dy) of the two partial derivative values.

    Example:
        >>> manual_partial_gradients(1.0, 2.0)
        (4.0, 5.0)
        # at (1, 2): df/dx = 4.0, df/dy = 5.0

    Hint:
        For df/dx: treat y as a constant and differentiate with respect to x.
        For df/dy: treat x as a constant and differentiate with respect to y.
    """
    raise NotImplementedError("Implement manual_partial_gradients()")


# ── Part 2: Autograd ──────────────────────────────────────────────────────────


def autograd_gradient_quadratic(a: float, b: float, x_val: float) -> float:
    """Return d/dx of f(x) = a*x**2 + b*x at x = x_val using torch.autograd.

    Args:
        a: Coefficient of the x**2 term.
        b: Coefficient of the x term.
        x_val: The point at which to evaluate the derivative.

    Returns:
        The value of f'(x_val) as a plain Python float.

    Example:
        >>> autograd_gradient_quadratic(1.0, 0.0, 3.0)
        6.0

    Hint:
        Create a tensor for x_val with requires_grad=True.
        Define f as a PyTorch expression. Call f.backward().
        The gradient lives in x.grad — return it as a Python float with .item().
    """
    raise NotImplementedError("Implement autograd_gradient_quadratic()")


def autograd_gradient_composite(x_val: float) -> float:
    """Return d/dx of f(x) = (x**2 + 1)**3 at x = x_val using torch.autograd.

    Args:
        x_val: The point at which to evaluate the derivative.

    Returns:
        The value of f'(x_val) as a plain Python float.

    Example:
        >>> autograd_gradient_composite(1.0)
        24.0

    Hint:
        Same pattern as autograd_gradient_quadratic — just define f differently.
    """
    raise NotImplementedError("Implement autograd_gradient_composite()")


def autograd_partial_gradients(x_val: float, y_val: float) -> tuple[float, float]:
    """Return (df/dx, df/dy) of f(x, y) = x**2 + x*y + y**2 using torch.autograd.

    Args:
        x_val: x-coordinate of the evaluation point.
        y_val: y-coordinate of the evaluation point.

    Returns:
        A tuple (df_dx, df_dy) as plain Python floats.

    Example:
        >>> autograd_partial_gradients(1.0, 2.0)
        (4.0, 5.0)

    Hint:
        Both x and y need requires_grad=True.
        After f.backward(), x.grad holds df/dx and y.grad holds df/dy.
    """
    raise NotImplementedError("Implement autograd_partial_gradients()")
