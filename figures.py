"""Génère images/fiabilite.png (nécessite matplotlib) : python figures.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyse import lire
from medpredict import ajuster_weibull, kaplan_meier, fiabilite, cout_horaire, intervalle_optimal, mtbf

heures, panne = lire("donnees/pousse-seringues.csv")
beta, eta = ajuster_weibull(heures, panne)
xs, ys = kaplan_meier(heures, panne)
t = np.linspace(0, 9000, 400)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
a1.step(np.r_[0, xs], np.r_[1, ys], where="post", label="Kaplan-Meier (historique)")
a1.plot(t, fiabilite(t, beta, eta), label=f"Weibull β={beta:.2f}, η={eta:.0f} h")
a1.set(xlabel="heures de fonctionnement", ylabel="fiabilité R(t)", title="Fiabilité du parc (données fictives)")
a1.grid(alpha=.3); a1.legend()
T = np.linspace(300, 9000, 300)
c = [cout_horaire(x, beta, eta, 150, 1200) * 1000 for x in T]
Topt, cT, sans = intervalle_optimal(beta, eta, 150, 1200)
a2.plot(T, c, label="avec remplacement préventif à l'âge T")
a2.axhline(sans * 1000, ls="--", c="gray", label="sans préventif")
a2.plot([Topt], [cT * 1000], "o", c="#c8102e", label=f"optimum : {Topt:.0f} h")
a2.set(xlabel="âge de remplacement T (h)", ylabel="€ / 1 000 h", title="Coût de maintenance (Cp 150 €, Cc 1 200 €)")
a2.grid(alpha=.3); a2.legend()
fig.tight_layout(); fig.savefig("images/fiabilite.png", dpi=130)
