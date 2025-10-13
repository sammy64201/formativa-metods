# UNIVERSIDAD DEL VALLE DE GUATEMALA
# EDGAR SAMUEL CHAVARRIA CASTAÑON 22055


# ------------------------------ LIBRERÍAS ------------------------------
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

# -------------------- FUNCIÓN OBJETIVO Y GRADIENTE ---------------------
def f(x, y):
    """Función a maximizar."""
    return np.sin(x) * np.cos(y) + (x**2 + y**2) / 10

def grad_f(x, y):
    """Gradiente de f(x,y)."""
    dfx = np.cos(x) * np.cos(y) + (2 * x) / 10
    dfy = -np.sin(x) * np.sin(y) + (2 * y) / 10
    return np.array([dfx, dfy])

# ----------------------- MÉTODO: PASO FIJO -----------------------------
def ascenso_paso_fijo(x0, alpha=0.05, tol=1e-4, max_iter=200):
    xk = np.array(x0, dtype=float)
    trayectoria = [xk.copy()]
    valores_f = [f(*xk)]
    for _ in range(max_iter):
        g = grad_f(*xk)
        if np.linalg.norm(g) < tol:
            break
        xk = xk + alpha * g
        trayectoria.append(xk.copy())
        valores_f.append(f(*xk))
    return np.array(trayectoria), np.array(valores_f)

# -------------------- BÚSQUEDAS LINEALES (1D) -------------------------
def _phi(alpha, xk, g):
    return f(*(xk + alpha * g))

def line_search_grid(xk, g, alpha_max=1.0, n=100):
    alphas = np.linspace(0.0, alpha_max, n)
    vals = [_phi(a, xk, g) for a in alphas]
    return float(alphas[int(np.argmax(vals))])

def line_search_golden(xk, g, a=0.0, b=1.0, tol=1e-5, max_iter=200):
    phi = lambda A: _phi(A, xk, g)
    gr = (np.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = phi(c)
    fd = phi(d)
    it = 0
    while (b - a) > tol and it < max_iter:
        if fc < fd:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = phi(d)
        else:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = phi(c)
        it += 1
    return float((a + b) / 2)

def line_search_newton(xk, g, alpha0=0.1, alpha_max=1.0, tol=1e-6, max_iter=50):
    def dphi(a, h=1e-4):
        return (_phi(a+h, xk, g) - _phi(a-h, xk, g)) / (2*h)
    def ddphi(a, h=1e-4):
        return (_phi(a+h, xk, g) - 2*_phi(a, xk, g) + _phi(a-h, xk, g)) / (h*h)

    a = float(np.clip(alpha0, 0.0, alpha_max))
    for _ in range(max_iter):
        g1, g2 = dphi(a), ddphi(a)
        if abs(g2) < 1e-12:
            break
        a_new = float(np.clip(a - g1 / g2, 0.0, alpha_max))
        if abs(a_new - a) < tol:
            return a_new
        a = a_new
    # Respaldo robusto si no convergió bien a un máximo
    return line_search_grid(xk, g, alpha_max=alpha_max, n=100)

# ------------- ASCENSO CON PASO ÓPTIMO (unificado) --------------------
def ascenso_paso_optimo(x0, tol=1e-4, max_iter=200, mode="golden",
                        alpha_max=1.0, n_grid=100, newton_alpha0=0.1):
    sel = (mode or "").lower().strip()
    if sel not in {"grid", "golden", "newton"}:
        raise ValueError("mode debe ser 'grid', 'golden' o 'newton'.")

    xk = np.array(x0, dtype=float)
    trayectoria = [xk.copy()]
    valores_f = [f(*xk)]
    alphas_usados = []

    for _ in range(max_iter):
        g = grad_f(*xk)
        if np.linalg.norm(g) < tol:
            break

        if sel == "grid":
            ak = line_search_grid(xk, g, alpha_max=alpha_max, n=n_grid)
        elif sel == "golden":
            ak = line_search_golden(xk, g, a=0.0, b=alpha_max, tol=1e-5, max_iter=200)
        else:  # 'newton'
            ak = line_search_newton(xk, g, alpha0=newton_alpha0, alpha_max=alpha_max, tol=1e-6, max_iter=50)

        alphas_usados.append(ak)
        xk = xk + ak * g
        trayectoria.append(xk.copy())
        valores_f.append(f(*xk))

    return np.array(trayectoria), np.array(valores_f), np.array(alphas_usados)

# --------------------------- AUXILIARES / PLOTS ------------------------
def graficar_contornos(tray_fijo, tray_opt=None, xlim=(-6,6), ylim=(-6,6), niveles=30, primeros_n=10):
    xx = np.linspace(xlim[0], xlim[1], 300)
    yy = np.linspace(ylim[0], ylim[1], 300)
    X, Y = np.meshgrid(xx, yy)
    Z = f(X, Y)
    plt.figure(figsize=(9,7))
    cs = plt.contour(X, Y, Z, levels=niveles, cmap="viridis")
    plt.clabel(cs, inline=1, fontsize=8, fmt="%.1f")

    plt.plot(tray_fijo[:,0], tray_fijo[:,1], "ro-", lw=1.5, ms=3, label="Paso fijo")
    for i in range(1, min(primeros_n, len(tray_fijo))):
        plt.annotate("", xy=tray_fijo[i], xytext=tray_fijo[i-1],
                     arrowprops=dict(arrowstyle="->", lw=1, color="r"))

    if tray_opt is not None:
        plt.plot(tray_opt[:,0], tray_opt[:,1], "bo-", lw=1.5, ms=3, label="Paso óptimo")
        for i in range(1, min(primeros_n, len(tray_opt))):
            plt.annotate("", xy=tray_opt[i], xytext=tray_opt[i-1],
                         arrowprops=dict(arrowstyle="->", lw=1, color="b"))

    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Curvas de nivel con trayectorias de iteraciones")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.show()

def graficar_superficie_3d_windowed(tray=None, xlim=(-6,6), ylim=(-6,6), titulo=""):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    xx = np.linspace(xlim[0], xlim[1], 150)
    yy = np.linspace(ylim[0], ylim[1], 150)
    X, Y = np.meshgrid(xx, yy)
    Z = f(X, Y)

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.75, linewidth=0, antialiased=True)

    if tray is not None:
        mask = ((tray[:,0] >= xlim[0]) & (tray[:,0] <= xlim[1]) &
                (tray[:,1] >= ylim[0]) & (tray[:,1] <= ylim[1]))
        tray_vis = tray[mask]
        if len(tray_vis) > 0:
            Zt = np.array([f(*p) for p in tray_vis])
            ax.plot(tray_vis[:,0], tray_vis[:,1], Zt, 'r.-', lw=2, label="Ruta")
            ax.legend()

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(np.min(Z), np.max(Z))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    ax.set_title(titulo or "Superficie 3D de f(x,y)")
    plt.show()

def grad_norm(xy):
    return float(np.linalg.norm(grad_f(*xy)))

def motivo_stop(xy_final, tol_grad):
    return "tol_grad" if grad_norm(xy_final) < tol_grad else "max_iter"

def imprimir_resultado(nombre, tray, vals, alpha_const=None, alphas_seq=None):
    if alpha_const is not None:
        alpha_txt = f"{alpha_const:.4f}"
    elif alphas_seq is not None and len(alphas_seq) > 0:
        alpha_txt = f"{alphas_seq[-1]:.4f}"
    else:
        alpha_txt = "—"
    print(f"{nombre}: x* = ({tray[-1,0]:.6f}, {tray[-1,1]:.6f}), f* = {vals[-1]:.6f}, alpha = {alpha_txt}")

def tabla_iteraciones(tray, vals, alphas=None, max_rows=None, titulo=""):
    n = len(tray)
    rows = n if max_rows is None else min(n, max_rows)
    if titulo:
        print(f"\n=== TABLA DE ITERACIONES ({titulo}) ===")
    if alphas is None:
        print("k |        x_k        |        y_k        |       f(x_k)")
        for k in range(rows):
            print(f"{k:3d}| {tray[k,0]:14.6f} | {tray[k,1]:14.6f} | {vals[k]:13.6f}")
    else:
        print("k |        x_k        |        y_k        |       f(x_k)    |  alpha_k")
        for k in range(rows):
            a = alphas[k-1] if k > 0 and (k-1) < len(alphas) else (alphas[k] if k < len(alphas) else 0.0)
            print(f"{k:3d}| {tray[k,0]:14.6f} | {tray[k,1]:14.6f} | {vals[k]:13.6f} | {a:8.5f}")

# ----------------------- RUTAS / CSV EN MISMA CARPETA -------------------
def _base_dir_usuario(base_dir=None):
    if base_dir is not None:
        return Path(base_dir).resolve()
    try:
        return Path(__file__).parent.resolve()  
    except NameError:
        return Path.cwd().resolve()          

# --------------- EXPORTA RESULTADOS A CSV (sin duplicidad) --------------
def correr_experimentos_csv(x0_list, alphas_fijos, tol, max_iter,
                            opt_mode="golden", alpha_max_opt=1.0, n_grid_opt=100, newton_alpha0=0.1,
                            csv_iter="iteraciones_todos.csv", csv_sum="resumen_experimentos.csv",
                            base_dir=None):
    base_path = _base_dir_usuario(base_dir)
    csv_iter_path = (base_path / csv_iter).resolve()
    csv_sum_path  = (base_path / csv_sum).resolve()

    iter_headers = ["metodo","x0_id","x0_x","x0_y","alpha_fijo","k","x_k","y_k","f_xk","alpha_k_opt"]
    sum_headers  = ["metodo","x0_id","x0_x","x0_y","alpha_fijo","k_final","x_final","y_final","f_final","grad_norm_final","motivo_stop"]

    # Crear/reescribir encabezados
    with csv_iter_path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(iter_headers)
    with csv_sum_path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(sum_headers)

    for i, x0 in enumerate(x0_list):
        x0 = np.array(x0, float)

        # ---- Paso fijo
        for a in alphas_fijos:
            tray, vals = ascenso_paso_fijo(x0, alpha=a, tol=tol, max_iter=max_iter)
            with csv_iter_path.open("a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                for k in range(len(tray)):
                    w.writerow(["paso_fijo", i, x0[0], x0[1], a, k, tray[k,0], tray[k,1], vals[k], ""])
            with csv_sum_path.open("a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(["paso_fijo", i, x0[0], x0[1], a,
                                         len(tray)-1, tray[-1,0], tray[-1,1], vals[-1],
                                         grad_norm(tray[-1]), motivo_stop(tray[-1], tol)])

        # ---- Paso óptimo (un solo modo elegido)
        tray_o, vals_o, alphas_o = ascenso_paso_optimo(
            x0, tol=tol, max_iter=max_iter, mode=opt_mode,
            alpha_max=alpha_max_opt, n_grid=n_grid_opt, newton_alpha0=newton_alpha0
        )
        metodo_tag = f"paso_optimo_{opt_mode}"
        with csv_iter_path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            for k in range(len(tray_o)):
                a_k = "" if k == 0 else (alphas_o[k-1] if (k-1) < len(alphas_o) else "")
                w.writerow([metodo_tag, i, x0[0], x0[1], "", k, tray_o[k,0], tray_o[k,1], vals_o[k], a_k])
        with csv_sum_path.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([metodo_tag, i, x0[0], x0[1], "",
                                     len(tray_o)-1, tray_o[-1,0], tray_o[-1,1], vals_o[-1],
                                     grad_norm(tray_o[-1]), motivo_stop(tray_o[-1], tol)])

    print(f"CSV de iteraciones: {csv_iter_path}")
    print(f"CSV de resumen:     {csv_sum_path}")
    return str(csv_iter_path), str(csv_sum_path)

# ---------------------- TEST RÁPIDO DE BÚSQUEDA ÓPTIMA ------------------
def probar_busqueda_optima(search_mode, x0_test, **kwargs):
    if search_mode is None or str(search_mode).strip().lower() == "null":
        return None
    tray, vals, alphas = ascenso_paso_optimo(x0_test, mode=search_mode, **kwargs)
    imprimir_resultado(f"BÚSQUEDA ÓPTIMA ({search_mode})", tray, vals, alpha_const=None, alphas_seq=alphas)
    tabla_iteraciones(tray, vals, alphas=alphas, max_rows=None, titulo=f"Óptimo ({search_mode})")
    return (tray, vals, alphas)

# =========================== CONFIGURACIÓN ==============================
# Puntos iniciales y parámetros
x0 = [-3.0, 3.0]  # para ejecuciones base/plots
x0_list = [(-3.0, 3.0), (-1.0, 4.0), (0.0, 0.0), (2.5, -0.5)]
alphas_fijos = [0.01, 0.05, 0.10]
tol = 1e-4
max_iter = 200

# =========================== EJECUCIONES ================================
# 1) Generar CSV (antes de graficar). Se guardan en la misma carpeta.
correr_experimentos_csv(
    x0_list, alphas_fijos, tol, max_iter,
    opt_mode="golden", alpha_max_opt=1.0, n_grid_opt=200, newton_alpha0=0.1
)

# 2) Ejecutar métodos base (para las figuras) e imprimir resultados
tray_fijo, val_fijo = ascenso_paso_fijo(x0, alpha=0.05, tol=tol, max_iter=max_iter)
tray_opt,  val_opt,  alphas = ascenso_paso_optimo(
    x0, tol=tol, max_iter=max_iter, mode="golden", alpha_max=1.0, n_grid=200, newton_alpha0=0.1
)
imprimir_resultado("PASO FIJO", tray_fijo, val_fijo, alpha_const=0.05, alphas_seq=None)
tabla_iteraciones(tray_fijo, val_fijo, alphas=None, max_rows=None, titulo="Paso fijo")

imprimir_resultado("PASO ÓPTIMO (golden)", tray_opt, val_opt, alpha_const=None, alphas_seq=alphas)
tabla_iteraciones(tray_opt, val_opt, alphas=alphas, max_rows=None, titulo="Paso óptimo (golden)")

# 3) (Opcional) Probar otro modo de búsqueda óptima
SEARCH_MODE = "grid"   # "grid" | "golden" | "newton" | None
_ = probar_busqueda_optima(
    SEARCH_MODE, x0_test=x0, alpha_max=1.0, n_grid=200,
    newton_alpha0=0.1, tol=tol, max_iter=max_iter
)

# 4) Gráficas
graficar_contornos(tray_fijo, tray_opt, xlim=(-6, 6), ylim=(-6, 6), niveles=30, primeros_n=10)
graficar_superficie_3d_windowed(tray=tray_fijo, xlim=(-6, 6), ylim=(-6, 6),
                                titulo="Superficie 3D + Ruta (Paso fijo)")
graficar_superficie_3d_windowed(tray=tray_opt, xlim=(-6, 6), ylim=(-6, 6),
                                titulo="Superficie 3D + Ruta (Paso óptimo)")
