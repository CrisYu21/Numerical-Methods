
import numpy as np


class MatrixOps:
    """
    A collection of common matrix operations.
    All inputs are accepted as Python lists or NumPy arrays and are
    converted internally to numpy float64 arrays.
    """

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _to_array(m) -> np.ndarray:
        """Convert list / nested list / ndarray → float64 ndarray."""
        return np.array(m, dtype=float)

    # ------------------------------------------------------------------
    # 1. Matrix Addition
    # ------------------------------------------------------------------

    @staticmethod
    def add(A, B) -> np.ndarray:
        """
        Element-wise addition: A + B.
        A and B must have the same shape.
        """
        A, B = MatrixOps._to_array(A), MatrixOps._to_array(B)
        if A.shape != B.shape:
            raise ValueError(
                f"Shape mismatch: A is {A.shape} but B is {B.shape}. "
                "Both matrices must have identical dimensions for addition."
            )
        return A + B

    # ------------------------------------------------------------------
    # 2. Matrix Multiplication
    # ------------------------------------------------------------------

    @staticmethod
    def multiply(A, B) -> np.ndarray:
        """
        Matrix product: A @ B.
        Number of columns of A must equal number of rows of B.
        """
        A, B = MatrixOps._to_array(A), MatrixOps._to_array(B)
        if A.ndim < 2 or B.ndim < 2:
            raise ValueError("Both inputs must be 2-D matrices.")
        if A.shape[1] != B.shape[0]:
            raise ValueError(
                f"Incompatible shapes for multiplication: "
                f"A is {A.shape}, B is {B.shape}. "
                f"A's column count ({A.shape[1]}) must equal B's row count ({B.shape[0]})."
            )
        return A @ B

    # ------------------------------------------------------------------
    # 3. Adjoint (Adjugate)
    # ------------------------------------------------------------------

    @staticmethod
    def adjoint(A) -> np.ndarray:
        """
        Adjugate (classical adjoint) of a square matrix.
        adj(A) = det(A) * A^{-1}  (when A is invertible).
        For singular matrices falls back to the cofactor transpose.
        """
        A = MatrixOps._to_array(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Adjoint requires a square 2-D matrix.")

        n   = A.shape[0]
        det = np.linalg.det(A)

        if abs(det) > 1e-10:
            # Fast path: adj(A) = det(A) * inv(A)
            return det * np.linalg.inv(A)

        # Cofactor matrix approach for singular matrices
        cofactors = np.zeros((n, n))
        for r in range(n):
            for c in range(n):
                minor = np.delete(np.delete(A, r, axis=0), c, axis=1)
                cofactors[r, c] = ((-1) ** (r + c)) * np.linalg.det(minor)
        return cofactors.T  # Adjugate = transpose of cofactor matrix

    # ------------------------------------------------------------------
    # 4. Inverse Matrix
    # ------------------------------------------------------------------

    @staticmethod
    def inverse(A) -> np.ndarray:
        """
        Matrix inverse A^{-1}.
        Raises ValueError for non-square matrices and
        np.linalg.LinAlgError if A is singular.
        """
        A = MatrixOps._to_array(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Inverse requires a square 2-D matrix.")
        det = np.linalg.det(A)
        if abs(det) < 1e-12:
            raise np.linalg.LinAlgError(
                f"Matrix is singular (det ≈ {det:.2e}); inverse does not exist."
            )
        return np.linalg.inv(A)

    # ------------------------------------------------------------------
    # 5. Determinant
    # ------------------------------------------------------------------

    @staticmethod
    def determinant(A) -> float:
        """
        Scalar determinant of a square matrix.
        """
        A = MatrixOps._to_array(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Determinant requires a square 2-D matrix.")
        return float(np.linalg.det(A))

    # ------------------------------------------------------------------
    # 6. Power of Matrix
    # ------------------------------------------------------------------

    @staticmethod
    def power(A, n: int) -> np.ndarray:
        """
        Integer matrix power A^n using np.linalg.matrix_power.
        n can be negative (requires invertible A).
        Works correctly for float matrices — does NOT cast to int.
        """
        A = MatrixOps._to_array(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix power requires a square 2-D matrix.")
        if not isinstance(n, int):
            raise TypeError(f"Power n must be an integer, got {type(n).__name__}.")
        if n < 0:
            det = np.linalg.det(A)
            if abs(det) < 1e-12:
                raise np.linalg.LinAlgError(
                    "Matrix is singular; negative power requires an invertible matrix."
                )
        return np.linalg.matrix_power(A, n)

    # ------------------------------------------------------------------
    # 7. Solve Linear Equations Ax = b
    # ------------------------------------------------------------------

    @staticmethod
    def solve(A, b) -> np.ndarray:
        """
        Solve the linear system Ax = b.
        Returns the solution vector x.
        Validates shapes and singularity before solving.
        """
        A = MatrixOps._to_array(A)
        b = MatrixOps._to_array(b).flatten()
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Coefficient matrix A must be square.")
        if A.shape[0] != b.shape[0]:
            raise ValueError(
                f"Dimension mismatch: A is {A.shape}, "
                f"b has {b.shape[0]} element(s). "
                f"A must have the same number of rows as elements in b."
            )
        det = np.linalg.det(A)
        if abs(det) < 1e-12:
            raise np.linalg.LinAlgError(
                f"Matrix A is singular (det ≈ {det:.2e}); system has no unique solution."
            )
        x        = np.linalg.solve(A, b)
        residual = np.linalg.norm(A @ x - b)
        if residual > 1e-6:
            raise RuntimeError(
                f"Solver residual is large (||Ax - b|| = {residual:.2e}). "
                "Check your inputs."
            )
        return x

    # ------------------------------------------------------------------
    # 8. Transpose
    # ------------------------------------------------------------------

    @staticmethod
    def transpose(A) -> np.ndarray:
        """
        Transpose of matrix A. Works for any 2-D matrix.
        """
        A = MatrixOps._to_array(A)
        if A.ndim != 2:
            raise ValueError("Transpose requires a 2-D matrix.")
        return A.T
