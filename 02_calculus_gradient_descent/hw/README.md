# HW02 — Derivatives and Autograd

This assignment has two parts that mirror each other:

- **Part 1 (Manual):** Implement three derivative computations using the rules
  you derived in lecture — power rule, chain rule, and partial derivatives.
  Pure Python arithmetic only.
- **Part 2 (Autograd):** Implement the exact same three computations using
  `torch.autograd`. Compare the results.

There is also a **math problem set** (separate PDF) covering hand-computed
derivatives, partial derivatives, gradient descent by hand, and a backprop
mini-problem. Submit that separately to Gradescope.

---

## Prerequisites

- HW01 completed (comfortable with PyTorch tensors)
- Attended the Calculus & Gradient Descent lecture (power rule, chain rule,
  partial derivatives, gradient, gradient descent update rule)

---

## Assignment structure

### Part 1 — Manual Derivatives (30 pts)

Implement the three functions in `derivatives.py` whose names start with
`manual_`. Use only Python arithmetic — no `torch`, no `math` library.

Functions to implement:
- `manual_gradient_quadratic` — power rule on $f(x) = ax^2 + bx$
- `manual_gradient_composite` — chain rule on $f(x) = (x^2 + 1)^3$
- `manual_partial_gradients` — partial derivatives of $f(x,y) = x^2 + xy + y^2$

### Part 2 — Autograd (30 pts)

Implement the three `autograd_*` functions in `derivatives.py`. These compute
the same derivatives as Part 1 using `torch.autograd`.

Functions to implement:
- `autograd_gradient_quadratic`
- `autograd_gradient_composite`
- `autograd_partial_gradients`

### Math Problem Set (graded separately)

A separate PDF accompanies this assignment. Complete all problems by hand,
scan or photograph your work, and upload as a PDF to Gradescope under
**HW02 — Math Problem Set**.

---

## Setup

Install dependencies (only needed once):

```bash
uv sync
```

---

## Running tests

Run the full test suite:

```bash
uv run pytest
```

Run only Part 1 (manual):

```bash
uv run pytest -m manual
```

Run only Part 2 (autograd):

```bash
uv run pytest -m autograd
```

---

## Checking your score

```bash
uv run python score.py           # full score
uv run python score.py manual    # Part 1 only
uv run python score.py autograd  # Part 2 only
```

Every test class is all-or-nothing: you earn full points for a class only
when **all** tests in that class pass.

---

## Submission

1. Zip your `hw/` folder. Your zip should contain only `derivatives.py`.
2. Upload the `.zip` to Gradescope under **HW02 — Derivatives (Coding)**.
3. Upload your math problem set PDF to Gradescope under
   **HW02 — Math Problem Set**.

Submit each part separately — two distinct Gradescope submissions.

---

## Notes on difficulty

Part 1 is mostly direct translation: once you know the derivative formula,
writing it in Python is straightforward. Part 2 is about learning the autograd
API — read the hints in the docstrings carefully.
