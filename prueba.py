# ================================================================
# ASCENSO DE MÁXIMA INCLINACIÓN – f2, tablas y gráficas (limpio)
# ================================================================

import numpy as np
import matplotlib.pyplot as plt

# -------------------- FUNCIÓN Y GRADIENTE -----------------------
def f(x, y):
    # f2(x,y) = sin(x)cos(y) + (x^2 + y^2)/10
    return np.sin(x) * np.cos(y) + (x**2 + y**2) / 10

def grad_f(x, y):
    dfx = np.cos(x) * np.cos(y) + (2 * x) / 10
    dfy = -np.sin(x) * np.sin(y) + (2 * y) / 10
    return np.array([dfx, dfy])

# ---------------- MÉTODO: PASO FIJO (SIN CAMBIOS) ----------------
def ascenso_paso_fijo(x0, alpha=0.05, tol=1e-4, max_iter=200):
    xk = np.array(x0, dtype=float)
    trayectoria = [xk.copy()]
    valores_f = [f(*xk)]
    for _ in range(max_iter):
        grad = grad_f(*xk)
        if np.linalg.norm(grad) < tol:
            break
        xk = xk + alpha * grad
        trayectoria.append(xk.copy())
        valores_f.append(f(*xk))
    return np.array(trayectoria), np.array(valores_f)

# -------------- MÉTODO: PASO ÓPTIMO (SIN CAMBIOS) ----------------
def ascenso_paso_optimo(x0, tol=1e-4, max_iter=200, alpha_max=1.0, n_busqueda=100):
    xk = np.array(x0, dtype=float)
    trayectoria = [xk.copy()]
    valores_f = [f(*xk)]
    alphas_usados = []
    for _ in range(max_iter):
        grad = grad_f(*xk)
        if np.linalg.norm(grad) < tol:
            break
        alphas = np.linspace(0, alpha_max, n_busqueda)
        candidatos = [f(*(xk + a * grad)) for a in alphas]
        a_opt = alphas[np.argmax(candidatos)]
        alphas_usados.append(a_opt)
        xk = xk + a_opt * grad
        trayectoria.append(xk.copy())
        valores_f.append(f(*xk))
    return np.array(trayectoria), np.array(valores_f), np.array(alphas_usados)

# ---------------------- TABLA DE ITERACIONES ---------------------
def tabla_iteraciones(tray, vals, alphas=None, max_rows=None, titulo=""):
    n = len(tray)
    rows = min(n, max_rows) if max_rows else n
    if titulo:
        print(f"\n=== TABLA DE ITERACIONES ({titulo}) ===")
    if alphas is None:
        print("k |        x_k        |        y_k        |       f(x_k)")
        for k in range(rows):
            print(f"{k:2d}| {tray[k,0]:14.6f} | {tray[k,1]:14.6f} | {vals[k]:13.6f}")
    else:
        # α_k usado para el paso k→k+1 (para k=0 no hay α previo)
        print("k |        x_k        |        y_k        |       f(x_k)    |  α_k")
        for k in range(rows):
            a = alphas[k-1] if (k>0 and (k-1)<len(alphas)) else (alphas[k] if k<len(alphas) else 0.0)
            print(f"{k:2d}| {tray[k,0]:14.6f} | {tray[k,1]:14.6f} | {vals[k]:13.6f} | {a:6.3f}")

# --------------------- CONTOUR CON RUTAS -------------------------
def graficar_contornos(tray_fijo, tray_opt=None, xlim=(-6,6), ylim=(-6,6), niveles=30, primeros_n=10):
    xx = np.linspace(xlim[0], xlim[1], 300)
    yy = np.linspace(ylim[0], ylim[1], 300)
    X, Y = np.meshgrid(xx, yy)
    Z = f(X, Y)

    plt.figure(figsize=(9,7))
    cs = plt.contour(X, Y, Z, levels=niveles, cmap="viridis")
    plt.clabel(cs, inline=1, fontsize=8, fmt="%.1f")

    # Ruta paso fijo
    plt.plot(tray_fijo[:,0], tray_fijo[:,1], "ro-", lw=1.5, ms=3, label="Paso fijo")
    for i in range(1, min(primeros_n, len(tray_fijo))):
        plt.annotate("", xy=tray_fijo[i], xytext=tray_fijo[i-1],
                     arrowprops=dict(arrowstyle="->", lw=1, color="r"))

    # Ruta paso óptimo
    if tray_opt is not None:
        plt.plot(tray_opt[:,0], tray_opt[:,1], "bo-", lw=1.5, ms=3, label="Paso óptimo")
        for i in range(1, min(primeros_n, len(tray_opt))):
            plt.annotate("", xy=tray_opt[i], xytext=tray_opt[i-1],
                         arrowprops=dict(arrowstyle="->", lw=1, color="b"))

    plt.xlim(*xlim); plt.ylim(*ylim)
    plt.xlabel("x"); plt.ylabel("y")
    plt.title("Curvas de nivel con trayectorias de iteraciones")
    plt.grid(True, alpha=0.25); plt.legend()
    plt.show()

# ----------------------- SUPERFICIE 3D ---------------------------
def graficar_superficie_3d_windowed(tray=None, xlim=(-6,6), ylim=(-6,6), titulo=""):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    # Malla fija para que la superficie siempre se vea
    xx = np.linspace(xlim[0], xlim[1], 150)
    yy = np.linspace(ylim[0], ylim[1], 150)
    X, Y = np.meshgrid(xx, yy)
    Z = f(X, Y)

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.75, linewidth=0, antialiased=True)

    # Recorte de la ruta a la ventana visible
    if tray is not None:
        mask = (
            (tray[:,0] >= xlim[0]) & (tray[:,0] <= xlim[1]) &
            (tray[:,1] >= ylim[0]) & (tray[:,1] <= ylim[1])
        )
        tray_vis = tray[mask]
        if len(tray_vis) > 0:
            Zt = np.array([f(*p) for p in tray_vis])
            ax.plot(tray_vis[:,0], tray_vis[:,1], Zt, 'r.-', lw=2, label="Ruta")
            ax.legend()

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_zlim(np.min(Z), np.max(Z))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("f(x,y)")
    ax.set_title(titulo or "Superficie 3D de f(x,y)")
    plt.show()

# ========================== EJECUCIÓN ============================
x0 = [-3.0, 3.0]

# 1) Correr métodos
tray_fijo, val_fijo = ascenso_paso_fijo(x0, alpha=0.05)
tray_opt,  val_opt,  alphas = ascenso_paso_optimo(x0, alpha_max=1.0, n_busqueda=100)

# 2) Resultados resumidos
print("=== PASO FIJO ===")
print(f"Punto final: ({tray_fijo[-1,0]:.6f}, {tray_fijo[-1,1]:.6f})  f = {val_fijo[-1]:.6f}")
print("=== PASO ÓPTIMO ===")
print(f"Punto final: ({tray_opt[-1,0]:.6f}, {tray_opt[-1,1]:.6f})  f = {val_opt[-1]:.6f}")

# 3) Tablas de iteraciones (recorta a 15 filas para consola)
tabla_iteraciones(tray_fijo, val_fijo, alphas=None,  max_rows=15, titulo="Paso fijo")
tabla_iteraciones(tray_opt,  val_opt,  alphas=alphas, max_rows=15, titulo="Paso óptimo")

# 4) Curvas de nivel con ambas trayectorias
graficar_contornos(tray_fijo, tray_opt, xlim=(-6,6), ylim=(-6,6), niveles=30, primeros_n=10)

# 5) Superficie 3D para cada método (ruta recortada a la ventana)
graficar_superficie_3d_windowed(tray=tray_fijo, xlim=(-6,6), ylim=(-6,6),
                                titulo="Superficie 3D + Ruta (Paso fijo)")
graficar_superficie_3d_windowed(tray=tray_opt,  xlim=(-6,6), ylim=(-6,6),
                                titulo="Superficie 3D + Ruta (Paso óptimo)")


# ================================================================
# EXPERIMENTOS: Inicialización (varios x0) y Parámetros (varios α)
# Exporta resultados a CSV y muestra un resumen en consola
# ================================================================

import csv
from math import sqrt

# --- Configuración de experimentos ---
x0_list = [
    (-3.0,  3.0),
    (-1.0,  4.0),
    ( 0.0,  0.0),
    ( 2.5, -0.5),
]
alphas_fijos = [0.01, 0.05, 0.10]  # para PASO FIJO
tol = 1e-4
max_iter = 200
alpha_max_opt = 1.0
n_busqueda_opt = 100

# --- Archivos de salida ---
csv_iter_path   = "iteraciones_todos.csv"
csv_resumen_path= "resumen_experimentos.csv"

# --- Helpers ---
def grad_norm(xy):
    g = grad_f(*xy)
    return float(np.linalg.norm(g))

def motivo_stop_desde_ultimo(xy_final, tol_grad):
    return "tol_grad" if grad_norm(xy_final) < tol_grad else "max_iter"

# encabezados CSV
iter_headers = [
    "metodo","x0_id","x0_x","x0_y","alpha_fijo",
    "k","x_k","y_k","f_xk","alpha_k_opt"
]
resumen_headers = [
    "metodo","x0_id","x0_x","x0_y","alpha_fijo",
    "k_final","x_final","y_final","f_final","grad_norm_final","motivo_stop"
]

# crear/limpiar archivos
with open(csv_iter_path, "w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh); writer.writerow(iter_headers)
with open(csv_resumen_path, "w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh); writer.writerow(resumen_headers)

# ------------------ Barrido de experimentos ---------------------
exp_id = 0
for i, x0 in enumerate(x0_list):
    x0 = np.array(x0, dtype=float)

    # ---- Método 1: PASO FIJO para cada α ----
    for a in alphas_fijos:
        tray, vals = ascenso_paso_fijo(x0, alpha=a, tol=tol, max_iter=max_iter)

        # Log iteraciones
        with open(csv_iter_path, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            for k in range(len(tray)):
                w.writerow(["paso_fijo", i, x0[0], x0[1], a,
                            k, tray[k,0], tray[k,1], vals[k], ""])

        # Resumen
        motivo = motivo_stop_desde_ultimo(tray[-1], tol)
        with open(csv_resumen_path, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["paso_fijo", i, x0[0], x0[1], a,
                        len(tray)-1, tray[-1,0], tray[-1,1], vals[-1],
                        grad_norm(tray[-1]), motivo])

    # ---- Método 2: PASO ÓPTIMO (una corrida por x0) ----
    tray_o, vals_o, alphas_o = ascenso_paso_optimo(
        x0, tol=tol, max_iter=max_iter,
        alpha_max=alpha_max_opt, n_busqueda=n_busqueda_opt
    )

    # Log iteraciones (α_k de la iteración k→k+1; α_0 = "")
    with open(csv_iter_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for k in range(len(tray_o)):
            alpha_k = "" if k == 0 else (alphas_o[k-1] if (k-1) < len(alphas_o) else "")
            w.writerow(["paso_optimo", i, x0[0], x0[1], "",
                        k, tray_o[k,0], tray_o[k,1], vals_o[k], alpha_k])

    motivo_o = motivo_stop_desde_ultimo(tray_o[-1], tol)
    with open(csv_resumen_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["paso_optimo", i, x0[0], x0[1], "",
                    len(tray_o)-1, tray_o[-1,0], tray_o[-1,1], vals_o[-1],
                    grad_norm(tray_o[-1]), motivo_o])

# ------------------ Mostrar un resumen corto --------------------
import pandas as pd
try:
    df_res = pd.read_csv(csv_resumen_path)
    print("\n=== Resumen por corrida ===")
    print(df_res.to_string(index=False))
    print(f"\n(Guardado: {csv_iter_path}, {csv_resumen_path})")
except Exception:
    # Fallback sin pandas
    print("\n(Resumen guardado en CSV)")
    with open(csv_resumen_path, "r", encoding="utf-8") as fh:
        print(fh.read())
