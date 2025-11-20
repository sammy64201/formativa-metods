"""
Simulación Monte Carlo:
Comparación de 3 estrategias de inversión
A: Ahorro en efectivo
B: Solo ETF
C: Portafolio combinado (efectivo + ETF)
"""

import numpy as np

# ==============================
# 1. VARIABLES GLOBALES
# ==============================

# Parámetros de simulación
AÑOS = 20
MESES_POR_AÑO = 12
N_MESES = AÑOS * MESES_POR_AÑO

N_SIMULACIONES = 10_000  # trayectorias simuladas
APORTE_MENSUAL = 1000.0  

# se inicia sin capital todas las simulaciones 
W0 = 0.0

# parametros de ETF
MU_ANUAL_ETF = 0.08      # el rendimiento anual
SIGMA_ANUAL_ETF = 0.15   # 1la volatilidad anual

# parametros de fondo de inversion
MU_ANUAL_CASH = 0.04     # 4% anual  segun el banco industrial
SIGMA_ANUAL_CASH = 0.005 

# conversiones
MU_MENSUAL_ETF = MU_ANUAL_ETF / MESES_POR_AÑO
SIGMA_MENSUAL_ETF = SIGMA_ANUAL_ETF / np.sqrt(MESES_POR_AÑO)

MU_MENSUAL_CASH = MU_ANUAL_CASH / MESES_POR_AÑO
SIGMA_MENSUAL_CASH = SIGMA_ANUAL_CASH / np.sqrt(MESES_POR_AÑO)

# ponderacion de la estrategia C
PESO_ETF_COMBINADO = 0.5  # 50/50

# meta de ahorro
META_FINAL = 500_000.0  

# Semilla para reproducibilidad
#SEED = 42


# ======================================================================================================================================================
# FUNCIONES
# ======================================================================================================================================================

# AHORRO EN FONDO DE INVERSION
def estrategia_efectivo(W0, aporte_mensual, retornos_cash):
    n_sim, n_meses = retornos_cash.shape
    riqueza = np.full(n_sim, W0, dtype=float)

    for k in range(n_meses):
        riqueza = (riqueza + aporte_mensual) * (1.0 + retornos_cash[:, k])

    return riqueza

# AHORRO EN INVERSION ETF
def estrategia_etf(W0, aporte_mensual, retornos_etf):
    n_sim, n_meses = retornos_etf.shape
    riqueza = np.full(n_sim, W0, dtype=float)

    for k in range(n_meses):
        riqueza = (riqueza + aporte_mensual) * (1.0 + retornos_etf[:, k])

    return riqueza

# COMBINACION DE ESTRATEGIAS
def estrategia_combinada(W0, aporte_mensual, retornos_etf, retornos_cash, peso_etf):
    n_sim, n_meses = retornos_etf.shape
    riqueza = np.full(n_sim, W0, dtype=float)


    retornos_comb = peso_etf * retornos_etf + (1.0 - peso_etf) * retornos_cash

    for k in range(n_meses):
        riqueza = (riqueza + aporte_mensual) * (1.0 + retornos_comb[:, k])

    return riqueza


# ======================================================================================================================================================
# SIMULACION DE MONTECARLO
# ======================================================================================================================================================

def ejecutar_simulaciones():
    rng = np.random.default_rng()

    #Generar retornos aleatorios para todas las simulaciones
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

    #Simular cada estrategia
    riqueza_final_A = estrategia_efectivo(W0, APORTE_MENSUAL, retornos_cash)
    riqueza_final_B = estrategia_etf(W0, APORTE_MENSUAL, retornos_etf)
    riqueza_final_C = estrategia_combinada(
        W0, APORTE_MENSUAL, retornos_etf, retornos_cash, PESO_ETF_COMBINADO
    )

    #Calcular estadísticas
    def stats(riqueza_final):
        media = np.mean(riqueza_final)
        mediana = np.median(riqueza_final)
        sigma = np.std(riqueza_final)
        p10 = np.percentile(riqueza_final, 10)
        p25 = np.percentile(riqueza_final, 25)
        p50 = np.percentile(riqueza_final, 50)
        p75 = np.percentile(riqueza_final, 75)
        p90 = np.percentile(riqueza_final, 90)
        prob_meta = np.mean(riqueza_final >= META_FINAL) 

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


if __name__ == "__main__":
    ejecutar_simulaciones()
