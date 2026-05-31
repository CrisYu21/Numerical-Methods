
import math


# ── Default sample function ───────────────────────────────────────────────────

def f(x: float) -> float:
    """f(x) = x³ − x − 2"""
    return x**3 - x - 2


def f_prime(x: float) -> float:
    """f′(x) = 3x² − 1"""
    return 3 * x**2 - 1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sign_str(product: float) -> str:
    """Return '> 0', '< 0', or '= 0' for a product value."""
    if product > 0:
        return "> 0"
    elif product < 0:
        return "< 0"
    return "= 0"


def _ea(x_new: float, x_old: float) -> float:
    """Approximate relative error in percent."""
    if x_new == 0:
        return float("inf")
    return abs((x_new - x_old) / x_new) * 100.0


def _fmt(v: float, digits: int = 6) -> float:
    """Round to `digits` decimal places for table display."""
    return round(v, digits)


# ── 1. Incremental Search ─────────────────────────────────────────────────────

def incremental_search(func, xl: float, xu: float,
                       dx: float = 0.1, tol: float = 1e-6):
    """
    Scan [xl, xu] in steps of dx, locating sign-change brackets.
    Table columns: Iteration | XL | Dx | XU | f(XL) | f(XU) | f(XL).f(XU) | Remark

    Returns (roots_list, table).
    The first sign-change bracket is refined via bisection to tolerance.
    """
    table  = []
    roots  = []
    x_cur  = xl
    f_cur  = func(x_cur)
    it     = 1

    while x_cur < xu:
        x_next = min(x_cur + dx, xu)
        f_next = func(x_next)
        prod   = f_cur * f_next

        if abs(prod) < 1e-14:
            prod_str = "= 0"
            remark   = "Exact root found"
            roots.append(x_next)
        elif prod < 0:
            prod_str = "< 0"
            remark   = "Revert back to XL & consider smaller interval"
            # Refine with bisection inside this bracket
            a_, b_ = x_cur, x_next
            for _ in range(200):
                m = (a_ + b_) / 2
                if abs(func(m)) < tol or (b_ - a_) / 2 < tol:
                    break
                if func(a_) * func(m) < 0:
                    b_ = m
                else:
                    a_ = m
            roots.append((a_ + b_) / 2)
        else:
            prod_str = "> 0"
            remark   = "Go to next interval"

        row = {
            "Iteration"   : it,
            "XL"          : _fmt(x_cur),
            "Dx"          : _fmt(dx),
            "XU"          : _fmt(x_next),
            "f(XL)"       : _fmt(f_cur),
            "f(XU)"       : _fmt(f_next),
            "f(XL).f(XU)" : prod_str,
            "Remark"      : remark,
        }
        table.append(row)

        x_cur = x_next
        f_cur = f_next
        it   += 1

    root = roots[0] if roots else None
    return root, table


# ── 2. Bisection ──────────────────────────────────────────────────────────────

def bisection(func, xl: float, xu: float,
              tol: float = 1e-6, max_iter: int = 100):
    """
    Table columns: Iteration | XL | XR | XU | f(XL) | f(XR) | f(XL).f(XR) | Ea(%) | Remark
    XR = (XL + XU) / 2
    Ea is '—' on iteration 1 (no prior estimate).
    """
    if func(xl) * func(xu) > 0:
        raise ValueError(
            f"f(XL) and f(XU) must have opposite signs.\n"
            f"f({xl}) = {func(xl):.6f},  f({xu}) = {func(xu):.6f}")

    table   = []
    xr_prev = None
    root    = None

    for i in range(1, max_iter + 1):
        xr   = (xl + xu) / 2
        f_xl = func(xl)
        f_xr = func(xr)
        prod = f_xl * f_xr

        # Ea: '—' on first iteration
        if xr_prev is None:
            ea_str = "—"
            ea_val = None
        else:
            ea_val = _ea(xr, xr_prev)
            ea_str = f"{ea_val:.6f}"

        prod_str = _sign_str(prod)

        if prod < 0:
            remark = "Root in left subinterval, XU = XR"
        elif prod > 0:
            remark = "Root in right subinterval, XL = XR"
        else:
            remark = "Exact root found"

        # Convergence checks
        if ea_val is not None and ea_val < tol * 100:
            remark = "Converged ✔"
        if abs(f_xr) < tol:
            remark = "Exact root found"

        row = {
            "Iteration"   : i,
            "XL"          : _fmt(xl),
            "XR"          : _fmt(xr),
            "XU"          : _fmt(xu),
            "f(XL)"       : _fmt(f_xl),
            "f(XR)"       : _fmt(f_xr),
            "f(XL).f(XR)" : prod_str,
            "Ea(%)"       : ea_str,
            "Remark"      : remark,
        }
        table.append(row)

        xr_prev = xr

        if remark in ("Exact root found", "Converged ✔"):
            root = xr
            break

        if prod < 0:
            xu = xr
        else:
            xl = xr

    root = root if root is not None else (xl + xu) / 2
    return root, table


# ── 3. Regula-Falsi (False Position) ─────────────────────────────────────────

def regula_falsi(func, xl: float, xu: float,
                 tol: float = 1e-6, max_iter: int = 100):
    """
    Table columns: Iteration | XL | XR | XU | f(XL) | f(XR) | f(XL).f(XR) | Ea(%) | Remark
    XR = XU - f(XU)*(XU - XL) / (f(XU) - f(XL))
    Ea is '—' on iteration 1 (no prior estimate).
    """
    if func(xl) * func(xu) > 0:
        raise ValueError(
            f"f(XL) and f(XU) must have opposite signs.\n"
            f"f({xl}) = {func(xl):.6f},  f({xu}) = {func(xu):.6f}")

    table   = []
    xr_prev = None
    root    = None

    for i in range(1, max_iter + 1):
        f_xl = func(xl)
        f_xu = func(xu)
        denom = f_xu - f_xl
        if abs(denom) < 1e-14:
            raise ZeroDivisionError("f(XU) - f(XL) ≈ 0; method cannot continue.")

        xr   = xu - f_xu * (xu - xl) / denom
        f_xr = func(xr)
        prod = f_xl * f_xr

        # Ea: '—' on first iteration
        if xr_prev is None:
            ea_str = "—"
            ea_val = None
        else:
            ea_val = _ea(xr, xr_prev)
            ea_str = f"{ea_val:.6f}"

        prod_str = _sign_str(prod)

        if prod < 0:
            remark = "Root in left subinterval, update XU = XR"
        elif prod > 0:
            remark = "Root in right subinterval, update XL = XR"
        else:
            remark = "Exact root found"

        if ea_val is not None and ea_val < tol * 100:
            remark = "Converged ✔"
        if abs(f_xr) < tol:
            remark = "Exact root found"

        row = {
            "Iteration"   : i,
            "XL"          : _fmt(xl),
            "XR"          : _fmt(xr),
            "XU"          : _fmt(xu),
            "f(XL)"       : _fmt(f_xl),
            "f(XR)"       : _fmt(f_xr),
            "f(XL).f(XR)" : prod_str,
            "Ea(%)"       : ea_str,
            "Remark"      : remark,
        }
        table.append(row)

        xr_prev = xr

        if remark in ("Exact root found", "Converged ✔"):
            root = xr
            break

        if prod < 0:
            xu = xr
        else:
            xl = xr

    root = root if root is not None else xr
    return root, table


# ── 4. Newton-Raphson ─────────────────────────────────────────────────────────

def newton_raphson(func, dfunc, x0: float,
                   tol: float = 1e-6, max_iter: int = 100):
    """
    Table columns: Iteration | Xi | f(Xi) | f'(Xi) | Xi+1 | Ea(%) | Remark
    Xi+1 = Xi - f(Xi) / f'(Xi)
    Ea is '—' on iteration 1 (no prior estimate).
    """
    table = []
    x     = x0
    root  = None

    for i in range(1, max_iter + 1):
        fx  = func(x)
        fpx = dfunc(x)

        if abs(fpx) < 1e-14:
            table.append({
                "Iteration": i,
                "Xi"       : _fmt(x),
                "f(Xi)"    : _fmt(fx),
                "f'(Xi)"   : _fmt(fpx),
                "Xi+1"     : "—",
                "Ea(%)"    : "—",
                "Remark"   : "Derivative is zero — method fails",
            })
            raise ZeroDivisionError(
                f"f′(Xi) ≈ 0 at iteration {i}. Newton-Raphson fails.")

        x_new = x - fx / fpx

        # Ea: '—' on first iteration (no prior estimate)
        if i == 1:
            ea_str = "—"
            ea_val = None
        else:
            ea_val = _ea(x_new, x)
            ea_str = f"{ea_val:.6f}"

        if ea_val is not None and ea_val < tol * 100:
            remark = "Converged ✔"
        elif abs(fx) < tol:
            remark = "Converged ✔"
        else:
            remark = "Converging"

        row = {
            "Iteration": i,
            "Xi"       : _fmt(x),
            "f(Xi)"    : _fmt(fx),
            "f'(Xi)"   : _fmt(fpx),
            "Xi+1"     : _fmt(x_new),
            "Ea(%)"    : ea_str,
            "Remark"   : remark,
        }
        table.append(row)

        x = x_new
        if remark == "Converged ✔":
            root = x
            break

    root = root if root is not None else x
    return root, table


# ── 5. Secant ─────────────────────────────────────────────────────────────────

def secant(func, x0: float, x1: float,
           tol: float = 1e-6, max_iter: int = 100):
    """
    Table columns: Iteration | Xi-1 | Xi | f(Xi-1) | f(Xi) | Xi+1 | Ea(%) | Remark
    Xi+1 = Xi - f(Xi)*(Xi - Xi-1) / (f(Xi) - f(Xi-1))
    Ea is '—' on iteration 1 (no prior estimate).
    """
    table = []
    root  = None

    for i in range(1, max_iter + 1):
        f0    = func(x0)
        f1    = func(x1)
        denom = f1 - f0

        if abs(denom) < 1e-14:
            table.append({
                "Iteration": i,
                "Xi-1"     : _fmt(x0),
                "Xi"       : _fmt(x1),
                "f(Xi-1)"  : _fmt(f0),
                "f(Xi)"    : _fmt(f1),
                "Xi+1"     : "—",
                "Ea(%)"    : "—",
                "Remark"   : "f(Xi) - f(Xi-1) ≈ 0 — method fails",
            })
            raise ZeroDivisionError(
                f"f(Xi) − f(Xi-1) ≈ 0 at iteration {i}. Secant method fails.")

        x2 = x1 - f1 * (x1 - x0) / denom

        # Ea: '—' on first iteration
        if i == 1:
            ea_str = "—"
            ea_val = None
        else:
            ea_val = _ea(x2, x1)
            ea_str = f"{ea_val:.6f}"

        if ea_val is not None and ea_val < tol * 100:
            remark = "Converged ✔"
        elif abs(f1) < tol:
            remark = "Converged ✔"
        else:
            remark = "Converging"

        row = {
            "Iteration": i,
            "Xi-1"     : _fmt(x0),
            "Xi"       : _fmt(x1),
            "f(Xi-1)"  : _fmt(f0),
            "f(Xi)"    : _fmt(f1),
            "Xi+1"     : _fmt(x2),
            "Ea(%)"    : ea_str,
            "Remark"   : remark,
        }
        table.append(row)

        x0, x1 = x1, x2
        if remark == "Converged ✔":
            root = x1
            break

    root = root if root is not None else x1
    return root, table
