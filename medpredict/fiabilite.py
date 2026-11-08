"""Fiabilité et maintenance de dispositifs médicaux (loi de Weibull).

Toutes les durées sont en heures de fonctionnement.
"""
import math
import numpy as np


# --- Loi de Weibull ---------------------------------------------------------

def fiabilite(t, beta, eta):
    """R(t) = exp(-(t/eta)^beta) : probabilité d'être encore en état à t."""
    return np.exp(-(np.asarray(t, float) / eta) ** beta)


def densite(t, beta, eta):
    t = np.asarray(t, float)
    return (beta / eta) * (t / eta) ** (beta - 1) * fiabilite(t, beta, eta)


def taux_defaillance(t, beta, eta):
    """h(t) = f(t)/R(t). beta < 1 : défauts de jeunesse, = 1 : aléatoires, > 1 : usure."""
    t = np.asarray(t, float)
    return (beta / eta) * (t / eta) ** (beta - 1)


def mtbf(beta, eta):
    """Durée moyenne de bon fonctionnement : eta * Gamma(1 + 1/beta)."""
    return eta * math.gamma(1 + 1 / beta)


def duree_bp(p, beta, eta):
    """Durée Bp : âge auquel p % des appareils sont tombés en panne (B10 pour p = 10)."""
    return eta * (-math.log(1 - p / 100)) ** (1 / beta)


def regime(beta):
    if beta < 0.95:
        return "défauts de jeunesse (taux de défaillance décroissant)"
    if beta <= 1.05:
        return "pannes aléatoires (taux constant)"
    return "usure (taux de défaillance croissant)"


# --- Estimation à partir des historiques -----------------------------------

def ajuster_weibull(durees, panne):
    """Maximum de vraisemblance avec données censurées à droite.

    durees : durées observées ; panne : True si l'appareil est tombé en panne,
    False s'il fonctionne encore (censure). Renvoie (beta, eta).
    """
    t = np.asarray(durees, float)
    d = np.asarray(panne, bool)
    r = d.sum()
    if r < 2:
        raise ValueError("Il faut au moins deux pannes observées.")
    lnt = np.log(t)
    moy_ln_pannes = lnt[d].mean()

    def g(b):  # dérivée de la log-vraisemblance profilée, décroissante en b
        tb = (t / t.max()) ** b  # mise à l'échelle pour éviter les débordements
        return (tb * lnt).sum() / tb.sum() - 1 / b - moy_ln_pannes

    lo, hi = 0.05, 50.0
    for _ in range(200):  # dichotomie : robuste, g change de signe une seule fois
        mid = 0.5 * (lo + hi)
        if g(mid) > 0:
            hi = mid
        else:
            lo = mid
    beta = 0.5 * (lo + hi)
    eta = t.max() * (((t / t.max()) ** beta).sum() / r) ** (1 / beta)
    return beta, eta


def kaplan_meier(durees, panne):
    """Estimateur empirique de R(t). Renvoie (instants de panne, R juste après)."""
    t = np.asarray(durees, float)
    d = np.asarray(panne, bool)
    ordre = np.lexsort((~d, t))  # à égalité, les pannes avant les censures
    t, d = t[ordre], d[ordre]
    n, s = len(t), 1.0
    xs, ys = [], []
    for i in range(n):
        if d[i]:
            s *= 1 - 1 / (n - i)
            xs.append(t[i])
            ys.append(s)
    return np.array(xs), np.array(ys)


# --- Maintenance préventive -------------------------------------------------

def _integrale_R(T, beta, eta, n=2000):
    T = min(T, eta * 40 ** (1 / beta))  # au-delà, R < e^-40 : contribution nulle
    x = np.linspace(0, T, n + 1)  # Simpson, n pair
    y = fiabilite(x, beta, eta)
    h = T / n
    return h / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum())


def cout_horaire(T, beta, eta, cout_preventif, cout_correctif):
    """Coût moyen par heure en remplaçant à l'âge T, ou à la panne si elle arrive avant.

    C(T) = [Cp R(T) + Cc (1 - R(T))] / intégrale de 0 à T de R(t) dt
    """
    R = float(fiabilite(T, beta, eta))
    return (cout_preventif * R + cout_correctif * (1 - R)) / _integrale_R(T, beta, eta)


def intervalle_optimal(beta, eta, cout_preventif, cout_correctif):
    """Âge de remplacement préventif qui minimise le coût horaire.

    Renvoie (T*, coût horaire à T*, coût horaire sans préventif) ou None
    si le préventif n'est pas rentable (beta <= 1 ou correctif pas plus cher).
    """
    sans = cout_correctif / mtbf(beta, eta)
    if beta <= 1 or cout_correctif <= cout_preventif:
        return None
    grille = np.linspace(0.02, 3, 300) * eta
    couts = [cout_horaire(T, beta, eta, cout_preventif, cout_correctif) for T in grille]
    i = int(np.argmin(couts))
    a, b = grille[max(i - 1, 0)], grille[min(i + 1, len(grille) - 1)]
    phi = (math.sqrt(5) - 1) / 2
    for _ in range(80):  # section dorée autour du minimum de la grille
        c, d = b - phi * (b - a), a + phi * (b - a)
        if cout_horaire(c, beta, eta, cout_preventif, cout_correctif) < cout_horaire(d, beta, eta, cout_preventif, cout_correctif):
            b = d
        else:
            a = c
    T = 0.5 * (a + b)
    cT = cout_horaire(T, beta, eta, cout_preventif, cout_correctif)
    if cT >= sans:
        return None
    return T, cT, sans


def disponibilite(mtbf_h, mttr_h):
    """Disponibilité intrinsèque A = MTBF / (MTBF + MTTR)."""
    return mtbf_h / (mtbf_h + mttr_h)
