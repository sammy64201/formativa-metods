# cuerda_clicks.py — Solo clics
# Dependencias: numpy, matplotlib
# pip install numpy matplotlib

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Si tu entorno no abre ventana, descomenta la siguiente línea ANTES de importar pyplot:
# matplotlib.use("TkAgg")

# Dimensiones reales (cm)
ANCHO_PEQUENO_CM = 201.8   # pizarrones izquierdo y derecho
ANCHO_GRANDE_CM  = 258.8   # pizarrón central

# ---------- Utilidades numéricas ----------
def integrar_escalado(P, regiones_x, escalas_cm_px):
    """Longitud total en cm asignando a cada segmento la escala según X medio."""
    P = np.asarray(P, float)
    if len(P) < 2: return 0.0
    total = 0.0
    for a, b in zip(P[:-1], P[1:]):
        midx = 0.5 * (a[0] + b[0])
        seg = np.linalg.norm(b - a)
        # región por punto medio
        idx = None
        for k, (xl, xr) in enumerate(regiones_x):
            if xl <= midx <= xr:
                idx = k; break
        if idx is None:  # región más cercana
            dmin, idx = 1e18, 0
            for k, (xl, xr) in enumerate(regiones_x):
                d = 0.0 if (xl <= midx <= xr) else min(abs(midx - xl), abs(midx - xr))
                if d < dmin: dmin, idx = d, k
        total += seg * escalas_cm_px[idx]
    return float(total)

def resample_by_arclength(P, step_px=1.5):
    """Remuestrea la polilínea cada ~step_px píxeles."""
    P = np.asarray(P, float)
    if len(P) < 2: return P
    segs = P[1:] - P[:-1]
    seglen = np.linalg.norm(segs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    L = s[-1]
    if L == 0: return P.copy()
    m = int(max(2, np.ceil(L / step_px)))
    target = np.linspace(0.0, L, m)
    Q = []; j = 0
    for t in target:
        while j < len(seglen) - 1 and s[j+1] < t: j += 1
        if seglen[j] == 0: Q.append(P[j].copy()); continue
        tau = (t - s[j]) / seglen[j]
        Q.append(P[j] * (1 - tau) + P[j+1] * tau)
    return np.asarray(Q)

def catmull_rom_chain(P, alpha=0.5, samples_per_seg=60):
    """
    Spline Catmull-Rom centrípeto (alpha≈0.5) sobre puntos P (Nx2).
    Sin SciPy. Devuelve puntos densos a lo largo de la curva.
    """
    P = np.asarray(P, float); n = len(P)
    if n < 2: return P.copy()
    if n == 2: return resample_by_arclength(P, 2.0)
    # duplicar extremos para condiciones de contorno
    Pext = np.vstack([P[0], P, P[-1]])
    Q = []
    def tj(ti, Pi, Pj): return ti + (np.linalg.norm(Pj - Pi) ** alpha)
    for i in range(1, n + 1):
        P0, P1, P2, P3 = Pext[i-1], Pext[i], Pext[i+1], Pext[i+2]
        t0 = 0.0
        t1 = tj(t0, P0, P1); t2 = tj(t1, P1, P2); t3 = tj(t2, P2, P3)
        ts = np.linspace(t1, t2, samples_per_seg, endpoint=(i == n))
        # Interpolación por segmentos
        A1 = (t1 - ts)[:, None] / (t1 - t0) * P0 + (ts - t0)[:, None] / (t1 - t0) * P1
        A2 = (t2 - ts)[:, None] / (t2 - t1) * P1 + (ts - t1)[:, None] / (t2 - t1) * P2
        A3 = (t3 - ts)[:, None] / (t3 - t2) * P2 + (ts - t2)[:, None] / (t3 - t2) * P3
        B1 = (t2 - ts)[:, None] / (t2 - t0) * A1 + (ts - t0)[:, None] / (t2 - t0) * A2
        B2 = (t3 - ts)[:, None] / (t3 - t1) * A2 + (ts - t1)[:, None] / (t3 - t1) * A3
        C  = (t2 - ts)[:, None] / (t2 - t1) * B1 + (ts - t1)[:, None] / (t2 - t1) * B2
        Q.append(C if i == n else C[:-1])
    return np.vstack(Q)

# ---------- Flujo principal (solo clics) ----------
def main():
    RUTA_IMAGEN = "pizarra.jpg"   # <--- cambia por el nombre real del archivo
    if not os.path.exists(RUTA_IMAGEN):
        print(f"[ERROR] No encontré '{RUTA_IMAGEN}'. Pon la imagen junto al script o da la ruta completa.")
        return

    img = plt.imread(RUTA_IMAGEN)

    # --- 1) Calibración: 6 clics ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(img)
    ax.set_title("Calibración: 6 clics (bordes izq/der) -> Peq Izq, Grande, Peq Der.\n"
                 "Cuando completes los 6 clics, cierra esta ventana o presiona Enter.")
    pts_cal = np.asarray(plt.ginput(n=6, timeout=0), float)
    plt.close(fig)

    if len(pts_cal) != 6:
        print(f"[ERROR] Se requieren 6 puntos de calibración; recibidos: {len(pts_cal)}")
        return

    def ordpar(par): return (min(par), max(par))
    x_peq_izq = ordpar((pts_cal[0,0], pts_cal[1,0]))
    x_grande  = ordpar((pts_cal[2,0], pts_cal[3,0]))
    x_peq_der = ordpar((pts_cal[4,0], pts_cal[5,0]))
    regiones_x = [x_peq_izq, x_grande, x_peq_der]

    anchos_px = [x_peq_izq[1]-x_peq_izq[0],
                 x_grande[1] -x_grande[0],
                 x_peq_der[1]-x_peq_der[0]]
    escalas = [ANCHO_PEQUENO_CM / max(anchos_px[0], 1e-9),
               ANCHO_GRANDE_CM  / max(anchos_px[1], 1e-9),
               ANCHO_PEQUENO_CM / max(anchos_px[2], 1e-9)]

    # --- 2) Cuerda: clics libres ---
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.imshow(img)
    # líneas guía (opcional)
    for (xl, xr), nom in zip(regiones_x, ["Peq Izq", "Grande", "Peq Der"]):
        ax2.axvline(x=xl, color='y', ls='--'); ax2.axvline(x=xr, color='y', ls='--')
        ax2.text((xl+xr)/2, 20, nom, color='y', ha='center',
                 bbox=dict(facecolor='black', alpha=0.3, pad=2))
    ax2.set_title("Clics sobre el centro de la cuerda (los que quieras).\n"
                  "Cierra la ventana o presiona Enter para terminar.")
    pts_cuerda = np.asarray(plt.ginput(n=-1, timeout=0), float)
    # dibujar lo que marcaste
    if len(pts_cuerda) >= 1:
        ax2.plot(pts_cuerda[:,0], pts_cuerda[:,1], 'r.-', lw=1, ms=3)
    plt.show()

    if len(pts_cuerda) < 2:
        print("[ERROR] Se necesitan al menos 2 puntos sobre la cuerda.")
        return

    # --- 3) Longitudes ---
    # Polilínea directa
    L_poly_cm = integrar_escalado(pts_cuerda, regiones_x, escalas)

    # Spline Catmull-Rom + remuestreo por arco (mejor aproximación)
    cr = catmull_rom_chain(pts_cuerda, alpha=0.5, samples_per_seg=60)
    cr_eq = resample_by_arclength(cr, step_px=1.5)
    L_spline_cm = integrar_escalado(cr_eq, regiones_x, escalas)

    L_prom = 0.5 * (L_poly_cm + L_spline_cm)

    # --- 4) Resultados ---
    print("\n=========== RESULTADOS ===========")
    print(f"Escalas cm/px -> Peq Izq: {escalas[0]:.6f} | Grande: {escalas[1]:.6f} | Peq Der: {escalas[2]:.6f}")
    print(f"Longitud (polilínea): {L_poly_cm:.2f} cm  ({L_poly_cm/100:.4f} m)")
    print(f"Longitud (spline)   : {L_spline_cm:.2f} cm  ({L_spline_cm/100:.4f} m)")
    print(f"Estimado recomendado: {L_prom:.2f} cm  ({L_prom/100:.4f} m)")
    print("==================================\n")

if __name__ == "__main__":
    main()
