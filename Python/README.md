# Numerical Methods 

A small GUI application (Tkinter) for exploring root-finding algorithms and common matrix operations.

## Features
- Part A — Root Finding: Incremental Search, Bisection, Regula-Falsi, Newton-Raphson, Secant
- Part B — Matrix Operations: Matrix Addition, Matrix Multiplication, Adjoint, Inverse Matrix, Determinants, Power of Matrix, Equations (Ax = b), Transpose of Matrix
- Interactive plotting for functions and a tabular iteration view for numerical methods
- Plain-text matrix input: one row per line, values space-separated; vectors are one value per line

## Requirements
- Python 3.8+
- Recommended packages:
  - numpy
  - matplotlib
  - sympy (optional — if not installed a numeric derivative fallback is used)

Install required packages with pip:

```bash
pip install numpy matplotlib sympy
```

Tkinter is included with most Python installations on Windows and many Linux distributions. If your system lacks Tkinter, install the appropriate system package (e.g., `sudo apt install python3-tk`).

## Run
From the project root directory run:

```bash
python main.py
```

## Matrix input format
- Matrix A / Matrix B: each row on its own line, values separated by spaces.
  Example 2x2 matrix:
  ```
  1 2
  3 4
  ```
- Vector `b` (for `Equations`): one value per line.

Note: By default the Matrix tab is now empty (no preloaded sample matrices). Use the `Clear` button to empty the fields at any time.

## Troubleshooting
- "Shape mismatch" errors mean your matrices have incompatible dimensions for the selected operation. Verify row/column counts match the operation's requirements.
- If the GUI fails to display, ensure `tkinter` and `matplotlib` backend dependencies are installed.

## Development
- Main application: `main.py`
- Matrix helper utilities: `matrix_ops.py`
- Numerical methods: `numerical_methods.py`

## License
This project is provided as-is.
