"""Vérifications : python tests/test_fiabilite.py"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from medpredict import (fiabilite, densite, taux_defaillance, mtbf, duree_bp, ajuster_weibull,
                        kaplan_meier, cout_horaire, intervalle_optimal, disponibilite)

n_ok = 0


def ok(cond, msg):
    global n_ok
    if not cond:
        raise AssertionError(msg)
    n_ok += 1


rng = np.random.default_rng(1)

# Formules de base
ok(abs(fiabilite(4200, 2.3, 4200) - math.exp(-1)) < 1e-12, "R(eta) = e^-1")
for beta in (0.7, 1.0, 2.5):
    t = np.linspace(1, 9000, 9)
    ok(np.allclose(taux_defaillance(t, beta, 3000), densite(t, beta, 3000) / fiabilite(t, beta, 3000)), "h = f / R")
    x = np.linspace(0, 3000 * 200, 2000001)
    num = np.trapezoid(fiabilite(x, beta, 3000), x)
    ok(abs(num - mtbf(beta, 3000)) / mtbf(beta, 3000) < 1e-3, f"MTBF = intégrale de R (beta {beta})")
ok(abs(mtbf(1, 500) - 500) < 1e-9, "beta = 1 : loi exponentielle, MTBF = eta")
ok(abs(float(fiabilite(duree_bp(10, 2.3, 4200), 2.3, 4200)) - 0.90) < 1e-12, "R(B10) = 90 %")

# Estimation sans censure : retrouve les paramètres sur un grand échantillon
for beta, eta in ((0.8, 1500), (1.5, 3000), (3.0, 6000)):
    t = eta * rng.weibull(beta, 20000)
    b, e = ajuster_weibull(t, np.ones_like(t, bool))
    ok(abs(b - beta) / beta < 0.03 and abs(e - eta) / eta < 0.03, f"MLE sans censure {beta}, {eta} -> {b:.3f}, {e:.0f}")

# Avec censure : ignorer les appareils encore en service fausserait le résultat
beta, eta = 2.3, 4200
vie = eta * rng.weibull(beta, 20000)
obs = rng.uniform(0, 9000, 20000)
t, d = np.minimum(vie, obs), vie <= obs
b, e = ajuster_weibull(t, d)
ok(abs(b - beta) / beta < 0.04 and abs(e - eta) / eta < 0.04, f"MLE avec censure -> {b:.3f}, {e:.0f}")
b_faux, e_faux = ajuster_weibull(t[d], np.ones(d.sum(), bool))
ok(e_faux < 0.9 * eta, "sans tenir compte de la censure, la durée de vie serait sous-estimée")

# Le maximum trouvé annule bien la dérivée de la vraisemblance
def logv(b_, e_):
    z = t / e_
    return (np.log(b_ / e_) + (b_ - 1) * np.log(z))[d].sum() - (z ** b_).sum()
for db in (-0.01, 0.01):
    ok(logv(b, e) >= logv(b + db, e) and logv(b, e) >= logv(b, e * (1 + db)), "maximum de vraisemblance")

# Kaplan-Meier sans censure = proportion empirique de survivants
t = np.array([5., 3, 9, 1, 7])
xs, ys = kaplan_meier(t, np.ones(5, bool))
ok(np.allclose(xs, [1, 3, 5, 7, 9]) and np.allclose(ys, [0.8, 0.6, 0.4, 0.2, 0.0]), "Kaplan-Meier sans censure")
xs, ys = kaplan_meier([2, 3, 4, 5], [True, False, True, True])
ok(np.allclose(ys, [0.75, 0.375, 0.0]), "Kaplan-Meier avec une censure")

# Maintenance préventive : section dorée = recherche exhaustive
for beta in (1.8, 2.3, 4.0):
    T, cT, sans = intervalle_optimal(beta, 4200, 150, 1200)
    grille = np.linspace(100, 12000, 6000)
    brute = min(cout_horaire(x, beta, 4200, 150, 1200) for x in grille)
    ok(abs(cT - brute) / brute < 1e-3 and cT < sans, f"optimum préventif beta {beta}")
ok(intervalle_optimal(1.0, 4200, 150, 1200) is None, "taux constant : le préventif ne sert à rien")
ok(intervalle_optimal(0.7, 4200, 150, 1200) is None, "défauts de jeunesse : le préventif aggrave")
ok(intervalle_optimal(2.3, 4200, 1200, 1200) is None, "préventif aussi cher que la panne : inutile")
ok(abs(cout_horaire(1e6, 2.3, 4200, 150, 1200) - 1200 / mtbf(2.3, 4200)) < 1e-6, "T très grand = tout en correctif")

ok(abs(disponibilite(990, 10) - 0.99) < 1e-12, "disponibilité")

print(f"{n_ok} vérifications réussies")
