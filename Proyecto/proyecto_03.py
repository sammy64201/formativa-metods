"""
Simulación Monte Carlo:
Comparación de 3 estrategias de inversión
A: Ahorro en efectivo
B: Solo ETF
C: Portafolio combinado (efectivo + ETF)
"""

import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1. VARIABLES GLOBALES
# ==============================

# Parámetros de simulación
AÑOS = 20
MESES_POR_AÑO = 12
N_MESES = AÑOS * MESES_POR_AÑO

N_SIMULACIONES = 10_000  # número de trayectorias (Monte Carlo)
APORTE_MENSUAL = 1000.0  # en la moneda que quieras (ej. Q)

# Capital inicial
W0 = 0.0

# Parámetros anuales del ETF (aproximados)
MU_ANUAL_ETF = 0.08      # 8% rendimiento esperado anual
SIGMA_ANUAL_ETF = 0.15   # 15% de volatilidad anual

# Parámetros anuales del efectivo (cuenta de ahorro / plazo fijo)
MU_ANUAL_CASH = 0.04     # 4% anual (según el banco industrial)
SIGMA_ANUAL_CASH = 0.005 # 0.5% anual (muy poca volatilidad)

# Conversión a parámetros mensuales (aprox)
MU_MENSUAL_ETF = MU_ANUAL_ETF / MESES_POR_AÑO
SIGMA_MENSUAL_ETF = SIGMA_ANUAL_ETF / np.sqrt(MESES_POR_AÑO)

MU_MENSUAL_CASH = MU_ANUAL_CASH / MESES_POR_AÑO
SIGMA_MENSUAL_CASH = SIGMA_ANUAL_CASH / np.sqrt(MESES_POR_AÑO)

# Peso del ETF en la estrategia combinada (C)
PESO_ETF_COMBINADO = 0.5  # 50% en ETF, 50% en efectivo

# Meta de riqueza al final del periodo (para calcular probabilidad de alcanzarla)
META_FINAL = 500_000.0  # ejemplo: 500,000 en la moneda que uses

# Si quieres simulación distinta cada vez, no uses seed fija:
# rng = np.random.default_rng()
# Aquí lo dejo sin SEED global a propósito
# SEED = 42


# ==============================
# 2. FUNCIONES POR ESTRATEGIA
# ==============================

def estrategia_efectivo(W0, aporte_mensual, retornos_cash):
    """
    Estrategia A: todo en efectivo (cash).
    retornos_cash: arreglo de shape (N_SIMULACIONES, N_MESES)
                   con los rendimientos mensuales aleatorios.
    Retorna: vector (N_SIMULACIONES,) con la riqueza final.
    """
    n_sim, n_meses = retornos_cash.shape
    riqueza = np.full(n_sim, W0, dtype=float)

    for k in range(n_meses):
        riqueza = (riqueza + aporte_mensual) * (1.0 + retornos_cash[:, k])

    return riqueza


def estrategia_etf(W0, aporte_mensual, retornos_etf):
    """
    Estrategia B: todo en ETF.
    retornos_etf: arreglo de shape (N_SIMULACIONES, N_MESES)
                  con los rendimientos mensuales aleatorios del ETF.
    Retorna: vector (N_SIMULACIONES,) con la riqueza final.
    """
    n_sim, n_meses = retornos_etf.shape
    riqueza = np.full(n_sim, W0, dtype=float)

    for k in range(n_meses):
        riqueza = (riqueza + aporte_mensual) * (1.0 + retornos_etf[:, k])

    return riqueza


def estrategia_combinada(W0, aporte_mensual, retornos_etf, retornos_cash, peso_etf):
    """
    Estrategia C: portafolio combinado.
    peso_etf: fracción invertida en ETF (el resto va a efectivo).
    retornos_etf y retornos_cash: shape (N_SIMULACIONES, N_MESES)
    Retorna: vector (N_SIMULACIONES,) con la riqueza final.
    """
    n_sim, n_meses = retornos_etf.shape
    riqueza = np.full(n_sim, W0, dtype=float)

    # rendimiento mensual del portafolio combinado
    retornos_comb = peso_etf * retornos_etf + (1.0 - peso_etf) * retornos_cash

    for k in range(n_meses):
        riqueza = (riqueza + aporte_mensual) * (1.0 + retornos_comb[:, k])

    return riqueza


# ==============================
# 3. FUNCIÓN PARA GRAFITOS
# ==============================

def graficar_resultados(riqueza_A, riqueza_B, riqueza_C, stats_A, stats_B, stats_C):
    """
    Genera y guarda figuras:
    - Histogramas por estrategia
    - Boxplot comparando A, B, C
    - Barras de probabilidad de alcanzar la meta
    """

    # --------- Histograma Estrategia A ---------
    plt.figure()
    plt.hist(riqueza_A, bins=50)
    plt.axvline(stats_A["media"], linestyle="--", label="Media")
    plt.title("Distribución capital final - Estrategia A (efectivo)")
    plt.xlabel("Capital final")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hist_estrategia_A.png", dpi=300)

    # --------- Histograma Estrategia B ---------
    plt.figure()
    plt.hist(riqueza_B, bins=50)
    plt.axvline(stats_B["media"], linestyle="--", label="Media")
    plt.title("Distribución capital final - Estrategia B (solo ETF)")
    plt.xlabel("Capital final")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hist_estrategia_B.png", dpi=300)

    # --------- Histograma Estrategia C ---------
    plt.figure()
    plt.hist(riqueza_C, bins=50)
    plt.axvline(stats_C["media"], linestyle="--", label="Media")
    plt.title("Distribución capital final - Estrategia C (combinado)")
    plt.xlabel("Capital final")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hist_estrategia_C.png", dpi=300)

    # --------- Boxplot comparativo ---------
    plt.figure()
    plt.boxplot(
        [riqueza_A, riqueza_B, riqueza_C],
        labels=["A: efectivo", "B: ETF", "C: combinado"]
    )
    plt.title("Comparación de capital final por estrategia")
    plt.ylabel("Capital final")
    plt.tight_layout()
    plt.savefig("boxplot_estrategias.png", dpi=300)

    # --------- Barras de probabilidad de alcanzar la meta ---------
    probs = [
        stats_A["prob_meta"] * 100,
        stats_B["prob_meta"] * 100,
        stats_C["prob_meta"] * 100,
    ]
    estrategias = ["A: efectivo", "B: ETF", "C: combinado"]

    plt.figure()
    plt.bar(estrategias, probs)
    plt.title(f"Probabilidad de alcanzar la meta de {META_FINAL:,.0f}")
    plt.ylabel("Probabilidad (%)")
    plt.tight_layout()
    plt.savefig("prob_meta_barras.png", dpi=300)

    # Si estás en Jupyter, esto muestra todas las figuras.
    # Si solo quieres guardarlas y no mostrarlas, puedes comentar la línea siguiente.
    plt.show()


# ==============================
# 4. FUNCIÓN PRINCIPAL DE MONTE CARLO
# ==============================

def ejecutar_simulaciones():
    """
    Ejecuta las simulaciones de Monte Carlo para las 3 estrategias,
    imprime resultados numéricos y genera gráficas.
    """

    # Si quieres que cada corrida sea distinta:
    rng = np.random.default_rng()
    # Si quisieras que siempre sea igual, usarías algo así:
    # rng = np.random.default_rng(42)

    # 1) Generar retornos aleatorios para todo el periodo y para todas las simulaciones
    retornos_etf = rng.normal(
        loc=MU_MENSUAL_ETF,
        scale=SIGMA_MENSUAL_ETF,
        size=(N_SIMULACIONES, N_MESES)
    )

    retornos_cash = rng.normal(
        loc=MU_MENSUAL_CASH,
        scale=SIGMA_MENSUAL_CASH,
        size=(N_SIMULACIONES, N_MESES)
    )

    # 2) Simular cada estrategia
    riqueza_final_A = estrategia_efectivo(W0, APORTE_MENSUAL, retornos_cash)
    riqueza_final_B = estrategia_etf(W0, APORTE_MENSUAL, retornos_etf)
    riqueza_final_C = estrategia_combinada(
        W0, APORTE_MENSUAL, retornos_etf, retornos_cash, PESO_ETF_COMBINADO
    )

    # 3) Calcular estadísticas
    def stats(riqueza_final):
        media = np.mean(riqueza_final)
        mediana = np.median(riqueza_final)
        sigma = np.std(riqueza_final)
        p10 = np.percentile(riqueza_final, 10)
        p25 = np.percentile(riqueza_final, 25)
        p50 = np.percentile(riqueza_final, 50)
        p75 = np.percentile(riqueza_final, 75)
        p90 = np.percentile(riqueza_final, 90)
        prob_meta = np.mean(riqueza_final >= META_FINAL)  # frecuencia relativa

        return {
            "media": media,
            "mediana": mediana,
            "sigma": sigma,
            "p10": p10,
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "p90": p90,
            "prob_meta": prob_meta
        }

    stats_A = stats(riqueza_final_A)
    stats_B = stats(riqueza_final_B)
    stats_C = stats(riqueza_final_C)

    # 4) Imprimir resultados numéricos
    print("===========================================")
    print(" PARÁMETROS DE LA SIMULACIÓN")
    print("===========================================")
    print(f"Años simulados           : {AÑOS}")
    print(f"Meses por año            : {MESES_POR_AÑO}")
    print(f"Número de simulaciones   : {N_SIMULACIONES}")
    print(f"Aporte mensual           : {APORTE_MENSUAL:.2f}")
    print(f"Capital inicial          : {W0:.2f}")
    print(f"Meta final               : {META_FINAL:.2f}")
    print()
    print("Rendimiento ETF anual    : {:.2%}".format(MU_ANUAL_ETF))
    print("Volatilidad ETF anual    : {:.2%}".format(SIGMA_ANUAL_ETF))
    print("Rendimiento cash anual   : {:.2%}".format(MU_ANUAL_CASH))
    print("Volatilidad cash anual   : {:.2%}".format(SIGMA_ANUAL_CASH))
    print(f"Peso ETF estrategia C    : {PESO_ETF_COMBINADO:.0%}")
    print("===========================================\n")

    def imprimir_stats(nombre, st):
        print(f"--- {nombre} ---")
        print("Media capital final        : {:.2f}".format(st["media"]))
        print("Mediana capital final      : {:.2f}".format(st["mediana"]))
        print("Desviación estándar        : {:.2f}".format(st["sigma"]))
        print("Percentil 10%              : {:.2f}".format(st["p10"]))
        print("Percentil 25%              : {:.2f}".format(st["p25"]))
        print("Percentil 50% (mediana)    : {:.2f}".format(st["p50"]))
        print("Percentil 75%              : {:.2f}".format(st["p75"]))
        print("Percentil 90%              : {:.2f}".format(st["p90"]))
        print("Prob. de alcanzar la meta  : {:.2%}".format(st["prob_meta"]))
        print()

    imprimir_stats("Estrategia A (solo efectivo)", stats_A)
    imprimir_stats("Estrategia B (solo ETF)", stats_B)
    imprimir_stats("Estrategia C (combinado)", stats_C)

    # 5) Generar y guardar gráficas
    graficar_resultados(
        riqueza_final_A, riqueza_final_B, riqueza_final_C,
        stats_A, stats_B, stats_C
    )


# ==============================
# 5. LLAMAR A LA FUNCIÓN PRINCIPAL
# ==============================

if __name__ == "__main__":
    ejecutar_simulaciones()
