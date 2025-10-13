# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ===========================
# ESTILO Y CONFIGURACIÓN UI
# ===========================
st.set_page_config(
    page_title="Ascenso de Máxima Inclinación – Interactivo",
    page_icon="📈",
    layout="wide"
)
st.markdown("""
<style>
/* Tipografía y espaciado */
html, body, [class*="css"]  { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, "Helvetica Neue", Arial; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
/* Cards */
.card { background: #111827; border-radius: 16px; padding: 16px 18px; border: 1px solid #1f2937; }
.metric { font-weight: 700; font-size: 1.15rem; }
.subtle { color: #9ca3af; font-size: 0.9rem; }
/* Buttons */
div.stDownloadButton > button, .stButton>button { border-radius: 10px; padding: 0.6rem 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ===========================
# FUNCIÓN OBJETIVO Y GRADIENTE
# ===========================
def f(x, y):
    return np.sin(x) * np.cos(y) + (x**2 + y**2) / 10.0

def grad_f(x, y):
    dfx = np.cos(x) * np.cos(y) + 0.2 * x
    dfy = -np.sin(x) * np.sin(y) + 0.2 * y
    return np.array([dfx, dfy], dtype=float)

# ===========================
# MÉTODOS: PASO FIJO / ÓPTIMO
# ===========================
def _phi(alpha, xk, g):
    return f(*(xk + alpha * g))

def line_search_grid(xk, g, alpha_max=1.0, n=100):
    alphas = np.linspace(0.0, alpha_max, n)
    vals = np.array([_phi(a, xk, g) for a in alphas])
    return float(alphas[int(np.argmax(vals))])

def line_search_golden(xk, g, a=0.0, b=1.0, tol=1e-5, max_iter=200):
    phi = lambda A: _phi(A, xk, g)
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a); d = a + gr * (b - a)
    fc = phi(c); fd = phi(d); it = 0
    while (b - a) > tol and it < max_iter:
        if fc < fd:
            a = c; c = d; fc = fd; d = a + gr*(b-a); fd = phi(d)
        else:
            b = d; d = c; fd = fc; c = b - gr*(b-a); fc = phi(c)
        it += 1
    return float((a + b) / 2.0)

def line_search_newton(xk, g, alpha0=0.1, alpha_max=1.0, tol=1e-6, max_iter=50):
    def dphi(a, h=1e-4):  return (_phi(a+h,xk,g)-_phi(a-h,xk,g))/(2*h)
    def ddphi(a,h=1e-4):  return (_phi(a+h,xk,g)-2*_phi(a,xk,g)+_phi(a-h,xk,g))/(h*h)
    a = float(np.clip(alpha0, 0.0, alpha_max))
    for _ in range(max_iter):
        g1, g2 = dphi(a), ddphi(a)
        if abs(g2) < 1e-12:
            break
        a_new = float(np.clip(a - g1/g2, 0.0, alpha_max))
        if abs(a_new - a) < tol:
            return a_new
        a = a_new
    # Respaldo robusto
    return line_search_grid(xk, g, alpha_max=alpha_max, n=100)

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
        else:
            ak = line_search_newton(xk, g, alpha0=newton_alpha0, alpha_max=alpha_max, tol=1e-6, max_iter=50)
        alphas_usados.append(ak)
        xk = xk + ak * g
        trayectoria.append(xk.copy())
        valores_f.append(f(*xk))
    return np.array(trayectoria), np.array(valores_f), np.array(alphas_usados)

# ===========================
# UTILIDADES
# ===========================
def build_contour_figure(tray_fijo, tray_opt, xlim, ylim, levels=40):
    xs = np.linspace(xlim[0], xlim[1], 300)
    ys = np.linspace(ylim[0], ylim[1], 300)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)

    fig = go.Figure()

    # Contornos
    fig.add_trace(go.Contour(
        z=Z, x=xs, y=ys, contours=dict(coloring='lines', showlabels=True),
        line=dict(width=1), showscale=False, name="Contornos"
    ))

    # Trayectoria paso fijo
    if tray_fijo is not None and len(tray_fijo) > 0:
        fig.add_trace(go.Scatter(
            x=tray_fijo[:,0], y=tray_fijo[:,1],
            mode="lines+markers", name="Paso fijo",
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}",
        ))

    # Trayectoria paso óptimo
    if tray_opt is not None and len(tray_opt) > 0:
        fig.add_trace(go.Scatter(
            x=tray_opt[:,0], y=tray_opt[:,1],
            mode="lines+markers", name="Paso óptimo",
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}",
        ))

    fig.update_layout(
        margin=dict(l=10, r=10, t=35, b=10),
        height=620,
        title="Curvas de nivel y trayectorias"
    )
    fig.update_xaxes(range=xlim, zeroline=True)
    fig.update_yaxes(range=ylim, zeroline=True, scaleanchor="x", scaleratio=1)
    return fig

def build_surface_3d(tray, xlim, ylim, title):
    xs = np.linspace(xlim[0], xlim[1], 120)
    ys = np.linspace(ylim[0], ylim[1], 120)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)

    fig = go.Figure(data=[
        go.Surface(x=xs, y=ys, z=Z, opacity=0.88, showscale=False)
    ])
    if tray is not None and len(tray) > 0:
        Zt = np.array([f(*p) for p in tray])
        fig.add_trace(go.Scatter3d(
            x=tray[:,0], y=tray[:,1], z=Zt,
            mode="lines+markers", name="Ruta", marker=dict(size=2)
        ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=35, b=10),
        height=520,
        title=title,
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="f(x,y)",
            xaxis=dict(range=xlim), yaxis=dict(range=ylim)
        )
    )
    return fig

def df_iters(method_name, x0, tray, vals, alphas=None):
    rows = []
    for k in range(len(tray)):
        rows.append({
            "metodo": method_name,
            "x0_x": x0[0],
            "x0_y": x0[1],
            "k": k,
            "x_k": tray[k,0],
            "y_k": tray[k,1],
            "f_xk": vals[k],
            "alpha_k_opt": (np.nan if (alphas is None or k==0 or (k-1)>=len(alphas)) else alphas[k-1])
        })
    return pd.DataFrame(rows)

def df_summary(method_name, x0, tray, vals, tol):
    g = grad_f(*tray[-1])
    motivo = "tol_grad" if np.linalg.norm(g) < tol else "max_iter"
    return pd.DataFrame([{
        "metodo": method_name,
        "x0_x": x0[0], "x0_y": x0[1],
        "k_final": len(tray)-1,
        "x_final": tray[-1,0], "y_final": tray[-1,1],
        "f_final": vals[-1],
        "grad_norm_final": np.linalg.norm(g),
        "motivo_stop": motivo
    }])

# ===========================
# SIDEBAR – CONTROLES
# ===========================
st.sidebar.header("Parámetros")
with st.sidebar:
    st.markdown("**Punto inicial**")
    colA, colB = st.columns(2)
    x0_x = colA.number_input("x0", value=-3.0, step=0.1, format="%.3f")
    x0_y = colB.number_input("y0", value= 3.0, step=0.1, format="%.3f")
    x0 = [x0_x, x0_y]

    st.markdown("---")
    st.markdown("**Dominio de visualización**")
    c1, c2 = st.columns(2)
    x_min = c1.number_input("x min", value=-6.0, step=0.5)
    x_max = c2.number_input("x max", value= 6.0, step=0.5)
    c3, c4 = st.columns(2)
    y_min = c3.number_input("y min", value=-6.0, step=0.5)
    y_max = c4.number_input("y max", value= 6.0, step=0.5)

    st.markdown("---")
    st.markdown("**Criterios de paro**")
    tol = st.number_input("tol (||∇f||)", value=1e-4, format="%.1e")
    max_iter = st.number_input("max_iter", min_value=1, value=200, step=10)

    st.markdown("---")
    st.markdown("**Método 1: Paso Fijo**")
    alpha_fijo = st.number_input("alpha (paso fijo)", value=0.05, step=0.01, format="%.3f")

    st.markdown("---")
    st.markdown("**Método 2: Paso Óptimo**")
    mode = st.selectbox("Búsqueda lineal", ("golden", "grid", "newton"))
    alpha_max = st.number_input("alpha_max", value=1.0, step=0.1, format="%.2f")
    n_grid = st.slider("n_grid (grid/respaldo)", min_value=20, max_value=2000, value=200, step=20)
    newton_alpha0 = st.number_input("newton_alpha0", value=0.1, step=0.05, format="%.2f")

    st.markdown("---")
    run = st.button("▶ Ejecutar simulación", use_container_width=True)

# ===========================
# CONTENIDO PRINCIPAL
# ===========================
st.title("Ascenso de Máxima Inclinación – Comparador Interactivo")
st.caption("Paso Fijo vs Paso Óptimo (Grid / Golden / Newton) con visualización en 2D/3D, métricas y exportación de resultados.")

if run:
    # Ejecutar Paso Fijo
    tray_fijo, val_fijo = ascenso_paso_fijo(x0, alpha=alpha_fijo, tol=tol, max_iter=int(max_iter))
    # Ejecutar Paso Óptimo
    tray_opt, val_opt, alphas = ascenso_paso_optimo(
        x0, tol=tol, max_iter=int(max_iter),
        mode=mode, alpha_max=alpha_max, n_grid=int(n_grid), newton_alpha0=newton_alpha0
    )

    # Métricas
    g_fijo = np.linalg.norm(grad_f(*tray_fijo[-1]))
    g_opt  = np.linalg.norm(grad_f(*tray_opt[-1]))
    stop_fijo = "tol_grad" if g_fijo < tol else "max_iter"
    stop_opt  = "tol_grad" if g_opt  < tol else "max_iter"

    # Header metrics
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.markdown(f"<div class='card'><div class='metric'>Paso Fijo</div><div class='subtle'>Iteraciones: <b>{len(tray_fijo)-1}</b></div><div class='subtle'>||∇f||: <b>{g_fijo:.3e}</b></div><div class='subtle'>f*: <b>{val_fijo[-1]:.6f}</b></div><div class='subtle'>Paro: <b>{stop_fijo}</b></div></div>", unsafe_allow_html=True)
    mcol2.markdown(f"<div class='card'><div class='metric'>Paso Óptimo ({mode})</div><div class='subtle'>Iteraciones: <b>{len(tray_opt)-1}</b></div><div class='subtle'>||∇f||: <b>{g_opt:.3e}</b></div><div class='subtle'>f*: <b>{val_opt[-1]:.6f}</b></div><div class='subtle'>Paro: <b>{stop_opt}</b></div></div>", unsafe_allow_html=True)
    mcol3.markdown(f"<div class='card'><div class='metric'>alpha fijo</div><div class='subtle'><b>{alpha_fijo:.4f}</b></div><div class='subtle'>alpha_max: <b>{alpha_max:.3f}</b></div><div class='subtle'>n_grid: <b>{int(n_grid)}</b></div></div>", unsafe_allow_html=True)
    if len(alphas)>0:
        mcol4.markdown(f"<div class='card'><div class='metric'>Último α óptimo</div><div class='subtle'><b>{alphas[-1]:.6f}</b></div><div class='subtle'>Total α_k: <b>{len(alphas)}</b></div></div>", unsafe_allow_html=True)
    else:
        mcol4.markdown(f"<div class='card'><div class='metric'>Último α óptimo</div><div class='subtle'><b>—</b></div><div class='subtle'>Total α_k: <b>0</b></div></div>", unsafe_allow_html=True)

    st.markdown("### Visualización 2D")
    fig2d = build_contour_figure(tray_fijo, tray_opt, (x_min, x_max), (y_min, y_max), levels=40)
    st.plotly_chart(fig2d, use_container_width=True, theme="streamlit")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### Superficie 3D – Paso Fijo")
        fig3d_fijo = build_surface_3d(tray_fijo, (x_min, x_max), (y_min, y_max), "Superficie 3D (Paso Fijo)")
        st.plotly_chart(fig3d_fijo, use_container_width=True, theme="streamlit")
    with c4:
        st.markdown("### Superficie 3D – Paso Óptimo")
        fig3d_opt = build_surface_3d(tray_opt, (x_min, x_max), (y_min, y_max), f"Superficie 3D (Paso Óptimo: {mode})")
        st.plotly_chart(fig3d_opt, use_container_width=True, theme="streamlit")

    # Gráfica f(x_k) vs k
    st.markdown("### Evolución de f(xₖ) por iteración")
    k_fijo = np.arange(len(val_fijo))
    k_opt  = np.arange(len(val_opt))
    fig_fxk = go.Figure()
    fig_fxk.add_trace(go.Scatter(x=k_fijo, y=val_fijo, mode="lines+markers", name="Paso Fijo"))
    fig_fxk.add_trace(go.Scatter(x=k_opt, y=val_opt, mode="lines+markers", name=f"Paso Óptimo ({mode})"))
    fig_fxk.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10), title="f(xₖ) vs iteración", xaxis_title="k", yaxis_title="f(xₖ)")
    st.plotly_chart(fig_fxk, use_container_width=True, theme="streamlit")

    # Tablas y descarga
    st.markdown("### Tablas y descarga de resultados")
    df_i_fijo = df_iters("paso_fijo", x0, tray_fijo, val_fijo, None)
    df_i_opt  = df_iters(f"paso_optimo_{mode}", x0, tray_opt, val_opt, alphas)
    df_s_fijo = df_summary("paso_fijo", x0, tray_fijo, val_fijo, tol)
    df_s_opt  = df_summary(f"paso_optimo_{mode}", x0, tray_opt, val_opt, tol)

    tabs = st.tabs(["Iteraciones – Paso Fijo", f"Iteraciones – Paso Óptimo ({mode})", "Resumen"])
    with tabs[0]:
        st.dataframe(df_i_fijo, use_container_width=True, hide_index=True)
        st.download_button("⬇ Descargar iteraciones (Paso Fijo)", data=df_i_fijo.to_csv(index=False).encode("utf-8"),
                           file_name="iteraciones_paso_fijo.csv", mime="text/csv")
    with tabs[1]:
        st.dataframe(df_i_opt, use_container_width=True, hide_index=True)
        st.download_button(f"⬇ Descargar iteraciones (Paso Óptimo – {mode})", data=df_i_opt.to_csv(index=False).encode("utf-8"),
                           file_name=f"iteraciones_paso_optimo_{mode}.csv", mime="text/csv")
    with tabs[2]:
        df_sum = pd.concat([df_s_fijo, df_s_opt], ignore_index=True)
        st.dataframe(df_sum, use_container_width=True, hide_index=True)
        st.download_button("⬇ Descargar resumen", data=df_sum.to_csv(index=False).encode("utf-8"),
                           file_name="resumen_experimentos.csv", mime="text/csv")

else:
    st.info("Ajusta los parámetros en la barra lateral y pulsa **“▶ Ejecutar simulación”** para empezar.")
    st.markdown("""
- **Elige el método**: Paso Fijo vs Paso Óptimo (Grid / Golden / Newton).
- **Controla el dominio** para ver mejor las rutas (x/y min/max).
- **Alpha fijo** controla el tamaño de paso constante.
- **Alpha_max** limita la búsqueda lineal (si f crece sin cota, αₖ puede pegarse al límite).
- **n_grid** define la resolución en rejilla (y respaldo de Newton).
- **Descarga** CSVs de iteraciones y resumen para tu informe.
""")
