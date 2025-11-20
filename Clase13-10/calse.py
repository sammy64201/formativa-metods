# Monte Carlo para integrar e^(x^2) en [a, b]
# Listo para ejecutar (solo requiere NumPy)

import numpy as np

def monte_carlo_integral(f, a=0.0, b=3.0, n_muestras=200_000, seed=42):
    """
    Estima ∫_a^b f(x) dx mediante muestreo uniforme.
    
    Parámetros
    ----------
    f : callable
        Función a integrar. Debe aceptar arrays de NumPy.
    a, b : float
        Límites de integración (a < b).
    n_muestras : int
        Número de muestras aleatorias (mayor -> menor varianza).
    seed : int or None
        Semilla para reproducibilidad. Usa None para aleatorio.

    Retorna
    -------
    estimacion : float
        Valor estimado de la integral.
    error_std : float
        Error estándar (desvío estándar del estimador).
    ci95 : tuple(float, float)
        Intervalo de confianza del 95% (inferior, superior).
    """
    if b <= a:
        raise ValueError("Se requiere b > a")
    rng = np.random.default_rng(seed)
    # Muestreo uniforme en [a,b]
    x = rng.uniform(a, b, size=n_muestras)
    fx = f(x)

    ancho = (b - a)
    media_fx = fx.mean()
    var_fx = fx.var(ddof=1)
    estimacion = ancho * media_fx
    # Varianza del estimador: (b-a)^2 * Var[f(X)] / n
    var_estimador = (ancho**2) * var_fx / n_muestras
    error_std = np.sqrt(var_estimador)
    z95 = 1.96
    ci95 = (estimacion - z95 * error_std, estimacion + z95 * error_std)
    return estimacion, error_std, ci95

def e_x2(x):
    """f(x) = e^(x^2) vectorizada."""
    return np.exp(x**2)

if __name__ == "__main__":
    # Ejemplo 1: integral en [0,1]
    est, err, (lo, hi) = monte_carlo_integral(e_x2, a=0.0, b=1.0, n_muestras=300_000, seed=123)
    print("∫_0^1 e^(x^2) dx ≈ {:.8f}  (±{:.8f}, 95% CI: [{:.8f}, {:.8f}])".format(est, err, lo, hi))

    # Ejemplo 2: integral en [-1,1]
    est2, err2, (lo2, hi2) = monte_carlo_integral(e_x2, a=-1.0, b=1.0, n_muestras=300_000, seed=123)
    print("∫_-1^1 e^(x^2) dx ≈ {:.8f}  (±{:.8f}, 95% CI: [{:.8f}, {:.8f}])".format(est2, err2, lo2, hi2))

    # Para cambiar precisión, aumenta n_muestras (p.ej., 1_000_000)
