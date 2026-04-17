# HW01 — Linear Algebra

In this assignment you will implement core linear algebra operations in two ways:
first from scratch in pure Python, then using PyTorch tensors.

By the end you will have built every fundamental building block — vector addition,
dot products, matrix multiplication, and solving linear systems — and you will
understand exactly what PyTorch is doing under the hood when it performs these
operations during model training.

---

## Prerequisites

- HW00 completed (comfortable with Python lists and functions)
- Attended the Linear Algebra lecture (vectors, matrices, dot products, matrix
  multiplication, linear systems)

---

## Assignment structure

### Part 1 — From Scratch (27 pts)

Implement the functions in `linear_algebra.py` using only Python built-ins and
the `math` module. No torch, no numpy. Vectors are `list[float]`; matrices are
`list[list[float]]` (stored in row-major order).

Functions to implement:
- `vector_add`
- `scalar_multiply`
- `dot_product`
- `vector_magnitude`
- `normalize_vector`
- `matrix_add`
- `matrix_vector_multiply`
- `matrix_multiply`
- `matrix_transpose`

### Part 2 — PyTorch Tensors (27 pts)

Implement more advanced operations using PyTorch. No manual loops — express
everything with PyTorch operators and tensor math. Each problem requires finding
the right primitive and combining it with shape manipulation or multi-step logic.

Functions to implement:
- `batch_dot`
- `polynomial_features`
- `row_normalize`
- `cosine_similarity`
- `pairwise_distances`
- `least_squares`

### Math Exercises (graded separately)

A separate PDF of written math exercises accompanies this assignment. Complete
them by hand and submit as a scanned or photographed PDF to Gradescope alongside
this coding assignment.

---

## Setup

Install dependencies (only needed once):

```bash
uv sync
```

(`uv` is a fast Python package manager — if you've used `pip` before, `uv sync` is equivalent to `pip install -r requirements.txt`.)

---

## Running tests

Run the full test suite:

```bash
uv run pytest
```

Run only Part 1:

```bash
uv run pytest -m scratch
```

Run only Part 2:

```bash
uv run pytest -m pytorch
```

---

## Checking your score

```bash
uv run python score.py          # full score
uv run python score.py scratch  # Part 1 only
uv run python score.py pytorch  # Part 2 only
```

Every test class is all-or-nothing: you earn full points for a class only when
**all** tests in that class pass. The score script shows which tests are failing
so you know what to fix.

---

## Submission

1. Zip your `hw/` folder (include `linear_algebra.py` and nothing else you
   should not modify).
2. Upload the `.zip` to Gradescope under **HW01 — Linear Algebra (Coding)**.
3. Upload your math exercises PDF to Gradescope under **HW01 — Linear Algebra
   (Math)**.

Submit each part separately — two distinct Gradescope submissions.

---

## Time estimate

**4–6 hours** for the coding portion. Budget additional time for the math
exercises.

Part 1 is mostly straightforward once you understand the definitions — the
challenge is translating the math notation into code precisely. Part 2 requires
reading the PyTorch documentation and thinking carefully about tensor shapes.
`least_squares` is worth the most points; read its docstring carefully.
