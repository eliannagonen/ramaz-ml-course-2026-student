"""Tests for HW01 — linear_algebra.py

Each class tests one function. All tests in a class must pass to earn credit.
Run with: uv run pytest
"""

from __future__ import annotations

import pytest
import torch

from linear_algebra import (
    batch_dot,
    cosine_similarity,
    dot_product,
    least_squares,
    matrix_add,
    matrix_multiply,
    matrix_transpose,
    matrix_vector_multiply,
    normalize_vector,
    pairwise_distances,
    polynomial_features,
    row_normalize,
    scalar_multiply,
    vector_add,
    vector_magnitude,
)

# ── Part 1: From Scratch ──────────────────────────────────────────────────────


@pytest.mark.scratch
class TestVectorAdd:
    def test_basic(self) -> None:
        result = vector_add([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert result == [5.0, 7.0, 9.0], (
            f"vector_add([1,2,3], [4,5,6]) should return [5,7,9]; got {result}"
        )

    def test_negative_values(self) -> None:
        result = vector_add([-1.0, -2.0], [1.0, 2.0])
        assert result == [0.0, 0.0], f"vector_add([-1,-2], [1,2]) should return [0,0]; got {result}"

    def test_single_element(self) -> None:
        result = vector_add([7.0], [3.0])
        assert result == [10.0], f"vector_add([7], [3]) should return [10]; got {result}"

    def test_returns_list(self) -> None:
        result = vector_add([1.0, 2.0], [3.0, 4.0])
        assert isinstance(result, list), f"Return type should be list, not {type(result).__name__}"

    def test_floats(self) -> None:
        result = vector_add([0.1, 0.2], [0.3, 0.4])
        assert abs(result[0] - 0.4) < 1e-6, f"0.1 + 0.3 should be close to 0.4; got {result[0]}"
        assert abs(result[1] - 0.6) < 1e-6, f"0.2 + 0.4 should be close to 0.6; got {result[1]}"


@pytest.mark.scratch
class TestScalarMultiply:
    def test_basic(self) -> None:
        result = scalar_multiply(3.0, [1.0, 2.0, 3.0])
        assert result == [3.0, 6.0, 9.0], (
            f"scalar_multiply(3, [1,2,3]) should return [3,6,9]; got {result}"
        )

    def test_zero_scalar(self) -> None:
        result = scalar_multiply(0.0, [1.0, 2.0, 3.0])
        assert result == [0.0, 0.0, 0.0], (
            f"Multiplying by 0 should return the zero vector; got {result}"
        )

    def test_negative_scalar(self) -> None:
        result = scalar_multiply(-1.0, [1.0, -2.0, 3.0])
        assert result == [-1.0, 2.0, -3.0], (
            f"scalar_multiply(-1, [1,-2,3]) should return [-1,2,-3]; got {result}"
        )

    def test_returns_list(self) -> None:
        result = scalar_multiply(2.0, [1.0, 2.0])
        assert isinstance(result, list), f"Return type should be list, not {type(result).__name__}"

    def test_fractional_scalar(self) -> None:
        result = scalar_multiply(0.5, [4.0, 8.0])
        assert abs(result[0] - 2.0) < 1e-6, f"0.5 * 4.0 should be 2.0; got {result[0]}"
        assert abs(result[1] - 4.0) < 1e-6, f"0.5 * 8.0 should be 4.0; got {result[1]}"


@pytest.mark.scratch
class TestDotProduct:
    def test_basic(self) -> None:
        result = dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert abs(result - 32.0) < 1e-6, f"dot([1,2,3], [4,5,6]) = 1*4+2*5+3*6 = 32; got {result}"

    def test_orthogonal_vectors(self) -> None:
        result = dot_product([1.0, 0.0], [0.0, 1.0])
        assert abs(result - 0.0) < 1e-6, f"Orthogonal unit vectors have dot product 0; got {result}"

    def test_parallel_unit_vectors(self) -> None:
        result = dot_product([1.0, 0.0], [1.0, 0.0])
        assert abs(result - 1.0) < 1e-6, f"Identical unit vectors have dot product 1; got {result}"

    def test_returns_float(self) -> None:
        result = dot_product([1.0, 2.0], [3.0, 4.0])
        assert isinstance(result, float), (
            f"Return type should be float, not {type(result).__name__}"
        )

    def test_negative_values(self) -> None:
        result = dot_product([-1.0, -2.0], [3.0, 4.0])
        assert abs(result - (-11.0)) < 1e-6, f"dot([-1,-2], [3,4]) = -3 + -8 = -11; got {result}"


@pytest.mark.scratch
class TestVectorMagnitude:
    def test_three_four_five(self) -> None:
        result = vector_magnitude([3.0, 4.0])
        assert abs(result - 5.0) < 1e-6, f"magnitude([3,4]) = sqrt(9+16) = 5; got {result}"

    def test_unit_vector(self) -> None:
        result = vector_magnitude([1.0, 0.0])
        assert abs(result - 1.0) < 1e-6, f"Unit vector [1,0] has magnitude 1; got {result}"

    def test_zero_vector(self) -> None:
        result = vector_magnitude([0.0, 0.0, 0.0])
        assert abs(result - 0.0) < 1e-6, f"Zero vector has magnitude 0; got {result}"

    def test_three_dimensions(self) -> None:
        result = vector_magnitude([1.0, 2.0, 2.0])
        assert abs(result - 3.0) < 1e-6, f"magnitude([1,2,2]) = sqrt(1+4+4) = 3; got {result}"

    def test_returns_float(self) -> None:
        result = vector_magnitude([3.0, 4.0])
        assert isinstance(result, float), (
            f"Return type should be float, not {type(result).__name__}"
        )


@pytest.mark.scratch
class TestNormalizeVector:
    def test_basic(self) -> None:
        result = normalize_vector([3.0, 4.0])
        assert abs(result[0] - 0.6) < 1e-6, f"normalize([3,4])[0] should be 0.6; got {result[0]}"
        assert abs(result[1] - 0.8) < 1e-6, f"normalize([3,4])[1] should be 0.8; got {result[1]}"

    def test_unit_vector_unchanged(self) -> None:
        result = normalize_vector([1.0, 0.0])
        assert abs(result[0] - 1.0) < 1e-6, (
            f"Normalizing a unit vector should leave it unchanged; got {result}"
        )
        assert abs(result[1] - 0.0) < 1e-6, (
            f"Normalizing a unit vector should leave it unchanged; got {result}"
        )

    def test_result_has_magnitude_one(self) -> None:
        result = normalize_vector([3.0, 4.0])
        mag = sum(x**2 for x in result) ** 0.5
        assert abs(mag - 1.0) < 1e-6, f"Normalized vector must have magnitude 1; got {mag}"

    def test_zero_vector_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_vector([0.0, 0.0, 0.0])

    def test_returns_list(self) -> None:
        result = normalize_vector([3.0, 4.0])
        assert isinstance(result, list), f"Return type should be list, not {type(result).__name__}"


@pytest.mark.scratch
class TestMatrixAdd:
    def test_basic(self) -> None:
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[5.0, 6.0], [7.0, 8.0]]
        result = matrix_add(A, B)
        assert result == [[6.0, 8.0], [10.0, 12.0]], (
            f"matrix_add([[1,2],[3,4]], [[5,6],[7,8]]) should return [[6,8],[10,12]]; got {result}"
        )

    def test_zero_matrix(self) -> None:
        A = [[1.0, 2.0], [3.0, 4.0]]
        Z = [[0.0, 0.0], [0.0, 0.0]]
        result = matrix_add(A, Z)
        assert result == [[1.0, 2.0], [3.0, 4.0]], (
            f"Adding a zero matrix should return the original matrix; got {result}"
        )

    def test_negative_values(self) -> None:
        A = [[1.0, -2.0], [-3.0, 4.0]]
        B = [[-1.0, 2.0], [3.0, -4.0]]
        result = matrix_add(A, B)
        assert result == [[0.0, 0.0], [0.0, 0.0]], (
            f"Negatives cancel; expected [[0,0],[0,0]]; got {result}"
        )

    def test_returns_list_of_lists(self) -> None:
        result = matrix_add([[1.0]], [[2.0]])
        assert isinstance(result, list) and isinstance(result[0], list), (
            f"Return type should be list[list[float]]; got {type(result)}"
        )

    def test_three_by_three(self) -> None:
        A = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        B = [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        result = matrix_add(A, B)
        assert result[0][1] == 1.0, f"result[0][1] should be 1.0; got {result[0][1]}"
        assert result[1][0] == 1.0, f"result[1][0] should be 1.0; got {result[1][0]}"


@pytest.mark.scratch
class TestMatrixVectorMultiply:
    def test_identity(self) -> None:
        identity = [[1.0, 0.0], [0.0, 1.0]]
        v = [3.0, 7.0]
        result = matrix_vector_multiply(identity, v)
        assert result == [3.0, 7.0], f"Identity matrix times v should return v; got {result}"

    def test_basic(self) -> None:
        A = [[1.0, 2.0], [3.0, 4.0]]
        v = [1.0, 1.0]
        result = matrix_vector_multiply(A, v)
        assert result == [3.0, 7.0], f"[[1,2],[3,4]] @ [1,1] = [1+2, 3+4] = [3,7]; got {result}"

    def test_two_by_three(self) -> None:
        A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        v = [1.0, 0.0, 1.0]
        result = matrix_vector_multiply(A, v)
        assert abs(result[0] - 4.0) < 1e-6, f"Row 0: 1*1+2*0+3*1 = 4; got {result[0]}"
        assert abs(result[1] - 10.0) < 1e-6, f"Row 1: 4*1+5*0+6*1 = 10; got {result[1]}"

    def test_output_length(self) -> None:
        A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        v = [1.0, 2.0, 3.0]
        result = matrix_vector_multiply(A, v)
        assert len(result) == 3, (
            f"3x3 matrix times 3-vector should give 3-vector; got length {len(result)}"
        )

    def test_returns_list(self) -> None:
        result = matrix_vector_multiply([[1.0, 0.0]], [5.0, 3.0])
        assert isinstance(result, list), f"Return type should be list, not {type(result).__name__}"


@pytest.mark.scratch
class TestMatrixMultiply:
    def test_two_by_two(self) -> None:
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[5.0, 6.0], [7.0, 8.0]]
        result = matrix_multiply(A, B)
        expected = [[19.0, 22.0], [43.0, 50.0]]
        assert result == expected, (
            f"[[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]; got {result}"
        )

    def test_identity(self) -> None:
        identity = [[1.0, 0.0], [0.0, 1.0]]
        A = [[3.0, 7.0], [1.0, 5.0]]
        result = matrix_multiply(identity, A)
        assert result == A, f"Identity @ A should return A; got {result}"

    def test_non_square(self) -> None:
        # A is 2x3, B is 3x2 -> result is 2x2
        # Expected: [[58, 64], [139, 154]]
        A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        B = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        result = matrix_multiply(A, B)
        expected = [[58.0, 64.0], [139.0, 154.0]]
        assert len(result) == 2, f"Result should have 2 rows; got {len(result)}"
        assert len(result[0]) == 2, f"Result should have 2 columns; got {len(result[0])}"
        for i in range(2):
            for j in range(2):
                assert abs(result[i][j] - expected[i][j]) < 1e-6, (
                    f"result[{i}][{j}] should be {expected[i][j]}; got {result[i][j]}"
                )

    def test_output_shape(self) -> None:
        # 3x2 times 2x4 -> 3x4
        A = [[float(i) for i in range(2)] for _ in range(3)]
        B = [[float(i) for i in range(4)] for _ in range(2)]
        result = matrix_multiply(A, B)
        assert len(result) == 3, f"Expected 3 rows; got {len(result)}"
        assert len(result[0]) == 4, f"Expected 4 cols; got {len(result[0])}"

    def test_returns_list_of_lists(self) -> None:
        result = matrix_multiply([[1.0]], [[2.0]])
        assert isinstance(result, list) and isinstance(result[0], list), (
            f"Return type should be list[list[float]]; got {type(result)}"
        )


@pytest.mark.scratch
class TestMatrixTranspose:
    def test_two_by_three(self) -> None:
        A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        result = matrix_transpose(A)
        expected = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        assert result == expected, f"Transpose of 2x3 matrix should be 3x2; got {result}"

    def test_square(self) -> None:
        A = [[1.0, 2.0], [3.0, 4.0]]
        result = matrix_transpose(A)
        assert result == [[1.0, 3.0], [2.0, 4.0]], (
            f"Transpose of [[1,2],[3,4]] = [[1,3],[2,4]]; got {result}"
        )

    def test_double_transpose_is_identity(self) -> None:
        A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        assert matrix_transpose(matrix_transpose(A)) == A, (
            "Transposing twice should give the original matrix"
        )

    def test_shape(self) -> None:
        A = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        result = matrix_transpose(A)
        assert len(result) == 4, f"Transposed 2x4 should have 4 rows; got {len(result)}"
        assert len(result[0]) == 2, f"Transposed 2x4 should have 2 cols; got {len(result[0])}"

    def test_returns_list_of_lists(self) -> None:
        result = matrix_transpose([[1.0, 2.0]])
        assert isinstance(result, list) and isinstance(result[0], list), (
            f"Return type should be list[list[float]]; got {type(result)}"
        )


# ── Part 2: PyTorch Tensors ───────────────────────────────────────────────────


@pytest.mark.pytorch
class TestBatchDot:
    def test_basic(self) -> None:
        U = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        V = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        result = batch_dot(U, V)
        expected = torch.tensor([17.0, 53.0])
        assert torch.allclose(result, expected, atol=1e-5), (
            f"batch_dot([[1,2],[3,4]], [[5,6],[7,8]]) = [1*5+2*6, 3*7+4*8] = [17,53]; got {result}"
        )

    def test_output_shape(self) -> None:
        U = torch.ones(5, 4)
        V = torch.ones(5, 4)
        result = batch_dot(U, V)
        assert result.shape == (5,), (
            f"batch_dot of (5,4) tensors should have shape (5,); got {result.shape}"
        )

    def test_orthogonal_rows(self) -> None:
        U = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        V = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        result = batch_dot(U, V)
        assert torch.allclose(result, torch.zeros(2), atol=1e-5), (
            f"Orthogonal row pairs should give dot products of 0; got {result}"
        )

    def test_matches_row_dots(self) -> None:
        # Verify against explicit dot products
        U = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        V = torch.tensor([[7.0, 8.0, 9.0], [1.0, 0.0, -1.0]])
        result = batch_dot(U, V)
        expected = torch.tensor(
            [
                1.0 * 7 + 2.0 * 8 + 3.0 * 9,  # 50
                4.0 * 1 + 5.0 * 0 + 6.0 * -1,  # -2
            ]
        )
        assert torch.allclose(result, expected, atol=1e-5), (
            f"Row-wise dot products: expected {expected}; got {result}"
        )

    def test_returns_tensor(self) -> None:
        result = batch_dot(torch.ones(2, 3), torch.ones(2, 3))
        assert isinstance(result, torch.Tensor), (
            f"Return type should be torch.Tensor, not {type(result).__name__}"
        )


@pytest.mark.pytorch
class TestPolynomialFeatures:
    def test_basic(self) -> None:
        x = torch.tensor([2.0, 3.0])
        result = polynomial_features(x, degree=2)
        expected = torch.tensor([[1.0, 2.0, 4.0], [1.0, 3.0, 9.0]])
        assert torch.allclose(result, expected, atol=1e-5), (
            f"polynomial_features([2,3], degree=2) should be [[1,2,4],[1,3,9]]; got {result}"
        )

    def test_degree_zero(self) -> None:
        x = torch.tensor([5.0, 7.0, 9.0])
        result = polynomial_features(x, degree=0)
        expected = torch.ones(3, 1)
        assert result.shape == (3, 1), f"degree=0 should give shape (3,1); got {result.shape}"
        assert torch.allclose(result, expected, atol=1e-5), (
            f"degree=0: every entry should be x^0=1; got {result}"
        )

    def test_output_shape(self) -> None:
        x = torch.arange(5, dtype=torch.float32)
        result = polynomial_features(x, degree=3)
        assert result.shape == (5, 4), (
            f"polynomial_features(x, degree=3) for len-5 x should have shape (5,4); "
            f"got {result.shape}"
        )

    def test_column_j_is_x_pow_j(self) -> None:
        x = torch.tensor([2.0, 3.0, 4.0])
        result = polynomial_features(x, degree=4)
        for j in range(5):
            expected_col = x**j
            assert torch.allclose(result[:, j], expected_col, atol=1e-5), (
                f"Column {j} should be x^{j}={expected_col}; got {result[:, j]}"
            )

    def test_returns_tensor(self) -> None:
        result = polynomial_features(torch.tensor([1.0, 2.0]), degree=2)
        assert isinstance(result, torch.Tensor), (
            f"Return type should be torch.Tensor, not {type(result).__name__}"
        )


@pytest.mark.pytorch
class TestRowNormalize:
    def test_rows_have_unit_norm(self) -> None:
        A = torch.tensor([[3.0, 4.0], [5.0, 12.0], [1.0, 0.0]])
        result = row_normalize(A)
        row_norms = torch.linalg.norm(result, dim=1)
        assert torch.allclose(row_norms, torch.ones(3), atol=1e-5), (
            f"Every row of the result should have norm 1; got row norms {row_norms}"
        )

    def test_basic_values(self) -> None:
        A = torch.tensor([[3.0, 4.0]])
        result = row_normalize(A)
        expected = torch.tensor([[0.6, 0.8]])
        assert torch.allclose(result, expected, atol=1e-5), (
            f"row_normalize([[3,4]]) should be [[0.6, 0.8]]; got {result}"
        )

    def test_shape_preserved(self) -> None:
        A = torch.randn(4, 5)
        result = row_normalize(A)
        assert result.shape == A.shape, (
            f"row_normalize should preserve shape; input {A.shape}, got {result.shape}"
        )

    def test_zero_row_raises(self) -> None:
        A = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
        with pytest.raises(ValueError):
            row_normalize(A)

    def test_returns_tensor(self) -> None:
        result = row_normalize(torch.tensor([[1.0, 0.0]]))
        assert isinstance(result, torch.Tensor), (
            f"Return type should be torch.Tensor, not {type(result).__name__}"
        )


@pytest.mark.pytorch
class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        u = torch.tensor([1.0, 2.0, 3.0])
        result = cosine_similarity(u, u)
        assert abs(result - 1.0) < 1e-5, (
            f"Cosine similarity of a vector with itself should be 1.0; got {result}"
        )

    def test_perpendicular_vectors(self) -> None:
        u = torch.tensor([1.0, 0.0])
        v = torch.tensor([0.0, 1.0])
        result = cosine_similarity(u, v)
        assert abs(result - 0.0) < 1e-5, (
            f"Perpendicular unit vectors should have cosine similarity 0; got {result}"
        )

    def test_opposite_vectors(self) -> None:
        u = torch.tensor([1.0, 2.0])
        v = torch.tensor([-1.0, -2.0])
        result = cosine_similarity(u, v)
        assert abs(result - (-1.0)) < 1e-5, (
            f"Opposite vectors should have cosine similarity -1; got {result}"
        )

    def test_known_value(self) -> None:
        # [1, 0] vs [1, 1]: angle = 45 degrees, cos(45) = 1/sqrt(2)
        u = torch.tensor([1.0, 0.0])
        v = torch.tensor([1.0, 1.0])
        result = cosine_similarity(u, v)
        expected = 1.0 / (2.0**0.5)
        assert abs(result - expected) < 1e-5, (
            f"cos_sim([1,0], [1,1]) = 1/sqrt(2) ≈ {expected:.4f}; got {result}"
        )

    def test_zero_vector_raises(self) -> None:
        with pytest.raises(ValueError):
            cosine_similarity(torch.zeros(3), torch.tensor([1.0, 2.0, 3.0]))

    def test_returns_python_float(self) -> None:
        u = torch.tensor([1.0, 0.0])
        v = torch.tensor([0.0, 1.0])
        result = cosine_similarity(u, v)
        assert isinstance(result, float), (
            f"Return type should be Python float; got {type(result).__name__}"
        )


@pytest.mark.pytorch
class TestPairwiseDistances:
    def test_known_values(self) -> None:
        A = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        result = pairwise_distances(A)
        expected = torch.tensor([[0.0, 5.0], [5.0, 0.0]])
        assert torch.allclose(result, expected, atol=1e-5), (
            f"pairwise_distances([[0,0],[3,4]]) should be [[0,5],[5,0]]; got {result}"
        )

    def test_diagonal_is_zero(self) -> None:
        A = torch.randn(4, 3)
        result = pairwise_distances(A)
        assert torch.allclose(result.diagonal(), torch.zeros(4), atol=1e-5), (
            f"Diagonal of pairwise_distances should be all zeros; got {result.diagonal()}"
        )

    def test_symmetry(self) -> None:
        A = torch.randn(5, 3)
        result = pairwise_distances(A)
        assert torch.allclose(result, result.mT, atol=1e-5), (
            "pairwise_distances must be symmetric: dist(i,j) == dist(j,i)"
        )

    def test_output_shape(self) -> None:
        A = torch.ones(6, 4)
        result = pairwise_distances(A)
        assert result.shape == (6, 6), (
            f"pairwise_distances of (6,4) tensor should have shape (6,6); got {result.shape}"
        )

    def test_returns_tensor(self) -> None:
        result = pairwise_distances(torch.eye(3))
        assert isinstance(result, torch.Tensor), (
            f"Return type should be torch.Tensor, not {type(result).__name__}"
        )


@pytest.mark.pytorch
class TestLeastSquares:
    def test_exact_fit(self) -> None:
        # Perfect linear data: y = 1 + 1*x
        X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        y = torch.tensor([2.0, 3.0, 4.0])
        w = least_squares(X, y)
        expected = torch.tensor([1.0, 1.0])
        assert torch.allclose(w, expected, atol=1e-4), (
            f"For y=1+x, least_squares should return [1,1]; got {w}"
        )

    def test_output_shape(self) -> None:
        X = torch.randn(10, 3)
        y = torch.randn(10)
        w = least_squares(X, y)
        assert w.shape == (3,), (
            f"least_squares with (10,3) X should return shape (3,); got {w.shape}"
        )

    def test_satisfies_normal_equations(self) -> None:
        # w minimizes ||Xw - y||^2, so X^T X w == X^T y
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = torch.tensor([1.0, 2.0, 3.0, 4.0])
        w = least_squares(X, y)
        lhs = X.mT @ X @ w
        rhs = X.mT @ y
        assert torch.allclose(lhs, rhs, atol=1e-4), (
            f"Solution should satisfy normal equations X^T X w = X^T y; lhs={lhs}, rhs={rhs}"
        )

    def test_predictions_close_to_targets(self) -> None:
        # Overdetermined but approximately linear data
        X = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        y = torch.tensor([0.5, 1.6, 2.4, 3.5])  # noisy y = 0.5 + x
        w = least_squares(X, y)
        predictions = X @ w
        residuals = (predictions - y).abs().max().item()
        assert residuals < 0.5, (
            f"Predictions should be close to targets for nearly-linear data; "
            f"max residual was {residuals:.4f}"
        )

    def test_returns_tensor(self) -> None:
        X = torch.eye(3)
        y = torch.tensor([1.0, 2.0, 3.0])
        result = least_squares(X, y)
        assert isinstance(result, torch.Tensor), (
            f"Return type should be torch.Tensor, not {type(result).__name__}"
        )
