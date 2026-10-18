import math
import numpy as np

# ============================================================
# Вариант 14
# u*(x,y) = x^2 y^2 (1 + y)
# L u = d/dx((3x+2) du/dx) + d^2u/dy^2
# L u = -f(x,y)
# ============================================================

def exact_u(x, y):
    return x * x * y * y * (1.0 + y)

def p_coef(x):
    return 3.0 * x + 2.0

def q_coef(x, y):
    return 1.0

def L_exact(x, y):
    # u = x^2(y^2+y^3)
    # L u = (12x+4)(y^2+y^3) + x^2(2+6y)
    return (12.0 * x + 4.0) * (y * y + y ** 3) + x * x * (2.0 + 6.0 * y)

def f_rhs_point(x, y):
    # L u = -f  =>  f = -L u
    return -L_exact(x, y)

# ============================================================
# Сетка и разностные операторы
# ============================================================

def make_grid(N):
    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N + 1)
    y = x.copy()
    X, Y = np.meshgrid(x, y, indexing="ij")
    return h, x, y, X, Y

def apply_boundary(U, N):
    # h = 1.0 / N
    x = np.linspace(0.0, 1.0, N + 1)
    y = x.copy()
    U[0, :] = exact_u(0.0, y)
    U[N, :] = exact_u(1.0, y)
    U[:, 0] = exact_u(x, 0.0)
    U[:, N] = exact_u(x, 1.0)
    return U

def zero_initial(N):
    U = np.zeros((N + 1, N + 1))
    apply_boundary(U, N)
    return U

def exact_grid(N):
    _, _, _, X, Y = make_grid(N)
    return exact_u(X, Y)

def rhs_grid(N):
    h = 1.0 / N
    F = np.zeros((N + 1, N + 1))
    for i in range(1, N):
        for j in range(1, N):
            F[i, j] = f_rhs_point(i * h, j * h)
    return F

def apply_Lh(U, N):
    h = 1.0 / N
    invh2 = 1.0 / (h * h)
    Lh = np.zeros_like(U)
    for i in range(1, N):
        xi = i * h
        pL = p_coef(xi - h / 2.0)
        pR = p_coef(xi + h / 2.0)
        for j in range(1, N):
            Lh[i, j] = (
                pR * (U[i + 1, j] - U[i, j]) * invh2
                - pL * (U[i, j] - U[i - 1, j]) * invh2
                + (U[i, j + 1] - U[i, j]) * invh2
                - (U[i, j] - U[i, j - 1]) * invh2
            )
    return Lh

def apply_Ax(U, N):
    h = 1.0 / N
    invh2 = 1.0 / (h * h)
    Ax = np.zeros_like(U)
    for i in range(1, N):
        xi = i * h
        pL = p_coef(xi - h / 2.0)
        pR = p_coef(xi + h / 2.0)
        for j in range(1, N):
            Ax[i, j] = ((pL + pR) * U[i, j] - pL * U[i - 1, j] - pR * U[i + 1, j]) * invh2
    return Ax

def apply_Ay(U, N):
    h = 1.0 / N
    invh2 = 1.0 / (h * h)
    Ay = np.zeros_like(U)
    for i in range(1, N):
        for j in range(1, N):
            Ay[i, j] = (2.0 * U[i, j] - U[i, j + 1] - U[i, j - 1]) * invh2
    return Ay

def norm_C(V):
    return float(np.max(np.abs(V[1:-1, 1:-1])))

# ============================================================
# Границы спектра и параметры
# ============================================================

def bounds_jacobi(N):
    h = 1.0 / N
    c1, c2 = 2.0, 5.0
    d1, d2 = 1.0, 1.0
    delta = (
        c1 * 4.0 / h ** 2 * math.sin(math.pi * h / 2.0) ** 2
        + d1 * 4.0 / h ** 2 * math.sin(math.pi * h / 2.0) ** 2
    )
    Delta = (
        c2 * 4.0 / h ** 2 * math.cos(math.pi * h / 2.0) ** 2
        + d2 * 4.0 / h ** 2 * math.cos(math.pi * h / 2.0) ** 2
    )
    return delta, Delta

def rho_jacobi(N):
    delta, Delta = bounds_jacobi(N)
    return (Delta - delta) / (Delta + delta)

def omega_sor(N):
    rho = rho_jacobi(N)
    return 2.0 / (1.0 + math.sqrt(max(0.0, 1.0 - rho * rho)))

def bounds_itm(N):
    h = 1.0 / N
    c1, c2 = 2.0, 5.0
    d1, d2 = 1.0, 1.0
    delta = (
        c1 * 4.0 / h ** 2 * math.sin(math.pi * h / 2.0) ** 2
        + d1 * 4.0 / h ** 2 * math.sin(math.pi * h / 2.0) ** 2
    )
    Delta = c2 * 4.0 / h ** 2 + d2 * 4.0 / h ** 2
    return delta, Delta

def estimate_iterations(rho, eps):
    if rho is None or rho <= 0.0 or rho >= 1.0:
        return float("nan")
    return int(math.ceil(math.log(eps) / math.log(rho)))

# ============================================================
# Метод 1. Итерация с оптимальным параметром
# ============================================================

def solve_optimal_parameter(N, eps=1e-3, max_iter=5000, callback=None):
    F = rhs_grid(N)
    U = zero_initial(N)
    U0 = U.copy()
    Uex = exact_grid(N)

    delta, Delta = bounds_jacobi(N)
    tau = 2.0 / (delta + Delta)

    for k in range(1, max_iter + 1):
        R = F + apply_Lh(U, N)      # F - A U, A = -Lh
        U = U + tau * R
        apply_boundary(U, N)

        if callback is not None:
            callback(k, U.copy())

        if norm_C(U - Uex) / norm_C(U0 - Uex) < eps:
            break
    return U

# ============================================================
# Метод 2. Зейдель
# ============================================================

def solve_seidel(N, eps=1e-3, max_iter=5000, callback=None):
    h = 1.0 / N
    invh2 = 1.0 / (h * h)
    F = rhs_grid(N)
    U = zero_initial(N)
    U0 = U.copy()
    Uex = exact_grid(N)

    for k in range(1, max_iter + 1):
        for i in range(1, N):
            xi = i * h
            pL = p_coef(xi - h / 2.0)
            pR = p_coef(xi + h / 2.0)
            denom = (pL + pR) * invh2 + 2.0 * invh2
            for j in range(1, N):
                num = (
                    pL * U[i - 1, j] * invh2
                    + pR * U[i + 1, j] * invh2
                    + U[i, j - 1] * invh2
                    + U[i, j + 1] * invh2
                    + F[i, j]
                )
                U[i, j] = num / denom

        if callback is not None:
            callback(k, U.copy())

        if norm_C(U - Uex) / norm_C(U0 - Uex) < eps:
            break
    return U

# ============================================================
# Метод 3. Верхняя релаксация
# ============================================================

def solve_sor(N, eps=1e-3, max_iter=5000, callback=None):
    h = 1.0 / N
    invh2 = 1.0 / (h * h)
    F = rhs_grid(N)
    U = zero_initial(N)
    U0 = U.copy()
    Uex = exact_grid(N)

    omega = omega_sor(N)
    if omega >= 2.0:
        omega = 1.95

    for k in range(1, max_iter + 1):
        for i in range(1, N):
            xi = i * h
            pL = p_coef(xi - h / 2.0)
            pR = p_coef(xi + h / 2.0)
            denom = (pL + pR) * invh2 + 2.0 * invh2
            for j in range(1, N):
                num = (
                    pL * U[i - 1, j] * invh2
                    + pR * U[i + 1, j] * invh2
                    + U[i, j - 1] * invh2
                    + U[i, j + 1] * invh2
                    + F[i, j]
                )
                gs = num / denom
                U[i, j] = (1.0 - omega) * U[i, j] + omega * gs

        if callback is not None:
            callback(k, U.copy())

        if norm_C(U - Uex) / norm_C(U0 - Uex) < eps:
            break
    return U

# ============================================================
# Вспомогательные решатели для попеременно-треугольного метода
# ============================================================

def solve_E_plus_omegaR(rhs, N, omega):
    h = 1.0 / N
    invh2 = 1.0 / (h * h)
    V = np.zeros_like(rhs)

    for i in range(1, N):
        xi = i * h
        pL = p_coef(xi - h / 2.0)
        den = 1.0 + omega * (pL * invh2 + invh2)
        for j in range(1, N):
            V[i, j] = (
                rhs[i, j]
                + omega * (pL * invh2 * V[i - 1, j] + invh2 * V[i, j - 1])
            ) / den
    return V

def solve_E_plus_omegaRT(rhs, N, omega):
    h = 1.0 / N
    invh2 = 1.0 / (h * h)
    V = np.zeros_like(rhs)

    for i in range(N - 1, 0, -1):
        xi = i * h
        pR = p_coef(xi + h / 2.0)
        den = 1.0 + omega * (pR * invh2 + invh2)
        for j in range(N - 1, 0, -1):
            V[i, j] = (
                rhs[i, j]
                + omega * (pR * invh2 * V[i + 1, j] + invh2 * V[i, j + 1])
            ) / den
    return V

# ============================================================
# Метод 4. Попеременно-треугольный с чебышевским набором
# ============================================================

def solve_itm_cheb(N, eps=1e-3, max_iter=5000, callback=None):
    F = rhs_grid(N)
    U = zero_initial(N)
    U0 = U.copy()
    Uex = exact_grid(N)

    delta, Delta = bounds_itm(N)
    eta = delta / Delta
    omega = 2.0 / math.sqrt(delta * Delta)
    gamma1 = delta / (2.0 + 2.0 * math.sqrt(eta))
    gamma2 = delta / (4.0 * math.sqrt(eta))

    n_cheb = 16
    # Порядок из методички для n=16
    order = [1, 31, 15, 17, 7, 25, 9, 23, 3, 29, 13, 19, 5, 27, 11, 21]

    for k in range(1, max_iter + 1):
        R = F + apply_Lh(U, N)

        wbar = solve_E_plus_omegaR(R, N, omega)
        w = solve_E_plus_omegaRT(wbar, N, omega)

        theta = order[(k - 1) % n_cheb]
        tau = 2.0 / (
            gamma2 + gamma1
            + (gamma2 - gamma1) * math.cos(theta * math.pi / (2.0 * n_cheb))
        )

        U = U + tau * w
        apply_boundary(U, N)

        if callback is not None:
            callback(k, U.copy())

        if norm_C(U - Uex) / norm_C(U0 - Uex) < eps:
            break
    return U

# ============================================================
# Метод 5. Метод переменных направлений (Peaceman-Rachford)
# ============================================================

def thomas(lower, diag, upper, b):
    n = len(diag)
    if n == 0:
        return np.array([])

    cp = upper.copy()
    dp = b.copy()
    dg = diag.copy()

    for i in range(1, n):
        m = lower[i] / dg[i - 1]
        dg[i] -= m * cp[i - 1]
        dp[i] -= m * dp[i - 1]

    x = np.zeros(n)
    x[-1] = dp[-1] / dg[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (dp[i] - cp[i] * x[i + 1]) / dg[i]
    return x

def adi_step_x(rhs, N, tau):
    h = 1.0 / N
    invh2 = 1.0 / (h * h)
    n = N - 1
    lower = np.zeros(n)
    diag = np.zeros(n)
    upper = np.zeros(n)

    for idx, i in enumerate(range(1, N)):
        xi = i * h
        pL = p_coef(xi - h / 2.0)
        pR = p_coef(xi + h / 2.0)
        lower[idx] = -tau * pL * invh2
        diag[idx] = 1.0 + tau * (pL + pR) * invh2
        upper[idx] = -tau * pR * invh2

    V = np.zeros_like(rhs)
    y = np.linspace(0.0, 1.0, N + 1)

    for j in range(N + 1):
        V[0, j] = exact_u(0.0, y[j])
        V[N, j] = exact_u(1.0, y[j])

    for j in range(1, N):
        b = rhs[1:N, j].copy()
        b[0] -= lower[0] * exact_u(0.0, y[j])
        b[-1] -= upper[-1] * exact_u(1.0, y[j])
        V[1:N, j] = thomas(lower, diag, upper, b)

    return V

def adi_step_y(rhs, N, tau):
    h = 1.0 / N
    invh2 = 1.0 / (h * h)
    n = N - 1
    lower = np.full(n, -tau * invh2)
    diag = np.full(n, 1.0 + 2.0 * tau * invh2)
    upper = np.full(n, -tau * invh2)

    V = np.zeros_like(rhs)
    x = np.linspace(0.0, 1.0, N + 1)

    for i in range(N + 1):
        V[i, 0] = exact_u(x[i], 0.0)
        V[i, N] = exact_u(x[i], 1.0)

    for i in range(1, N):
        b = rhs[i, 1:N].copy()
        b[0] -= lower[0] * exact_u(x[i], 0.0)
        b[-1] -= upper[-1] * exact_u(x[i], 1.0)
        V[i, 1:N] = thomas(lower, diag, upper, b)

    return V

def solve_adi(N, eps=1e-3, max_iter=5000, callback=None):
    F = rhs_grid(N)
    U = zero_initial(N)
    U0 = U.copy()
    Uex = exact_grid(N)

    delta, Delta = bounds_itm(N)
    m_adi = 8
    taus = np.geomspace(1.0 / Delta, 1.0 / delta, m_adi)

    for k in range(1, max_iter + 1):
        tau = taus[(k - 1) % m_adi]

        Ay = apply_Ay(U, N)
        rhs_half = U - tau * Ay + tau * F
        Uhalf = adi_step_x(rhs_half, N, tau)

        Ax = apply_Ax(Uhalf, N)
        rhs_new = Uhalf - tau * Ax + tau * F
        Unew = adi_step_y(rhs_new, N, tau)

        U = Unew

        if callback is not None:
            callback(k, U.copy())

        if norm_C(U - Uex) / norm_C(U0 - Uex) < eps:
            break
    return U

# ============================================================
# Вывод характеристик по п. 8.7
# ============================================================

def print_table(history, N, F, Uex, U0, rho_theory, every=1):
    base_res = norm_C(F + apply_Lh(U0, N))
    base_err = norm_C(U0 - Uex)

    print("\nТаблица характеристик:")
    print(
        f"{'k':>5} {'||F-AU^k||':>14} {'rel.res':>12} "
        f"{'||U^k-u*||':>14} {'rel.err':>12} "
        f"{'||U^k-U^(k-1)||':>18} {'apost.est':>14} {'rho_k':>10}"
    )

    prev = U0
    prev_diff = None

    for idx, U in enumerate(history, start=1):
        R = F + apply_Lh(U, N)
        res = norm_C(R)
        rel_res = res / base_res if base_res > 0 else float("nan")

        err = norm_C(U - Uex)
        rel_err = err / base_err if base_err > 0 else float("nan")

        diff = norm_C(U - prev)
        rho_k = diff / prev_diff if (prev_diff is not None and prev_diff > 0.0) else float("nan")

        if rho_theory is not None and 0.0 < rho_theory < 1.0:
            apost = rho_theory / (1.0 - rho_theory) * diff
        else:
            apost = float("nan")

        if idx == 1 or idx % every == 0 or idx == len(history):
            print(
                f"{idx:5d} {res:14.6e} {rel_res:12.4e} "
                f"{err:14.6e} {rel_err:12.4e} "
                f"{diff:18.6e} {apost:14.6e} {rho_k:10.4e}"
            )

        prev = U
        prev_diff = diff

def print_coarse(U, N, title):
    step = N // 5
    if step < 1:
        step = 1
    idx = np.arange(0, N + 1, step)
    xvals = np.linspace(0.0, 1.0, N + 1)[idx]

    print(f"\n{title} на крупной сетке:")
    print("y\\x  " + " ".join(f"{v:9.4f}" for v in xvals))
    for j in idx:
        y = j / N
        row = [U[i, j] for i in idx]
        print(f"{y:4.2f} " + " ".join(f"{v:9.5f}" for v in row))

def print_characteristics(method_name, N, U, history, rho_theory, eps):
    F = rhs_grid(N)
    Uex = exact_grid(N)
    U0 = zero_initial(N)

    approx_measure = norm_C(F + apply_Lh(Uex, N))
    res0 = norm_C(F + apply_Lh(U0, N))

    m_est = estimate_iterations(rho_theory, eps)

    if len(history) >= 3:
        d1 = norm_C(history[-1] - history[-2])
        d2 = norm_C(history[-2] - history[-3])
        rho_exp = d1 / d2 if d2 > 0.0 else float("nan")
    else:
        rho_exp = float("nan")

    print("\n" + "=" * 90)
    print(f"Метод: {method_name}")
    print(f"N = {N}, eps = {eps}")
    print(f"1) Мера аппроксимации ||F - A u*|| = {approx_measure:.6e}")
    print(f"2) Норма невязки нулевого приближения ||F - A U0|| = {res0:.6e}")
    print(f"3) Оценка числа итераций m_est = {m_est}, фактически выполнено = {len(history)}")
    print(f"4) Теоретический spectral radius rho_H = {rho_theory}")
    print(f"   Экспериментальная оценка rho_exp = {rho_exp:.6e}")

    every = 1 if N == 5 else 10
    print_table(history, N, F, Uex, U0, rho_theory, every=every)

    print_coarse(U, N, "Приближённое решение U^m")
    print_coarse(Uex, N, "Точное решение u*")

# ============================================================
# Основная программа
# ============================================================

def main():
    eps = 1e-3
    max_iter = 5000

    for N in [5, 10, 20]:
        print("\n\n" + "#" * 100)
        print(f"# Сетка N = {N}")
        print("#" * 100)

        # 1. Метод итерации с оптимальным параметром
        hist = []
        U = solve_optimal_parameter(
            N, eps=eps, max_iter=max_iter,
            callback=lambda k, Uk: hist.append(Uk.copy())
        )
        rho = rho_jacobi(N)
        print_characteristics(
            "Метод итерации с оптимальным параметром",
            N, U, hist, rho, eps
        )

        # 2. Метод Зейделя
        hist = []
        U = solve_seidel(
            N, eps=eps, max_iter=max_iter,
            callback=lambda k, Uk: hist.append(Uk.copy())
        )
        rho = rho_jacobi(N) ** 2
        print_characteristics(
            "Метод Зейделя",
            N, U, hist, rho, eps
        )

        # 3. Метод верхней релаксации
        hist = []
        U = solve_sor(
            N, eps=eps, max_iter=max_iter,
            callback=lambda k, Uk: hist.append(Uk.copy())
        )
        rho = omega_sor(N) - 1.0
        print_characteristics(
            "Метод верхней релаксации",
            N, U, hist, rho, eps
        )

        # 4. Попеременно-треугольный с чебышевским набором
        hist = []
        U = solve_itm_cheb(
            N, eps=eps, max_iter=max_iter,
            callback=lambda k, Uk: hist.append(Uk.copy())
        )
        print_characteristics(
            "Попеременно-треугольный метод с чебышевским набором",
            N, U, hist, None, eps
        )

        # 5. Метод переменных направлений
        hist = []
        U = solve_adi(
            N, eps=eps, max_iter=max_iter,
            callback=lambda k, Uk: hist.append(Uk.copy())
        )
        print_characteristics(
            "Метод переменных направлений",
            N, U, hist, None, eps
        )

main()
