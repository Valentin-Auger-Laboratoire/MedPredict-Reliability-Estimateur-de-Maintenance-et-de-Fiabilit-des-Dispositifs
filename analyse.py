"""Rapport de fiabilité d'un parc d'appareils à partir d'un historique CSV.

Usage : python analyse.py donnees/pousse-seringues.csv [--cp 150] [--cc 1200] [--mttr 6]
Le CSV contient : appareil, heures, etat (panne ou en_service).
"""
import argparse
import csv

from medpredict import (ajuster_weibull, mtbf, duree_bp, regime, fiabilite,
                        intervalle_optimal, disponibilite)


def lire(chemin):
    heures, panne = [], []
    with open(chemin, newline="", encoding="utf-8") as f:
        for ligne in csv.DictReader(f):
            heures.append(float(ligne["heures"]))
            panne.append(ligne["etat"].strip() == "panne")
    return heures, panne


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv")
    p.add_argument("--cp", type=float, default=150, help="coût d'un remplacement préventif (€)")
    p.add_argument("--cc", type=float, default=1200, help="coût d'une panne en service (€)")
    p.add_argument("--mttr", type=float, default=6, help="durée moyenne de réparation (h)")
    a = p.parse_args()

    heures, panne = lire(a.csv)
    beta, eta = ajuster_weibull(heures, panne)
    m = mtbf(beta, eta)
    print(f"Appareils : {len(heures)} dont {sum(panne)} pannes et {len(heures) - sum(panne)} encore en service")
    print(f"Weibull : beta = {beta:.2f}, eta = {eta:.0f} h -> {regime(beta)}")
    print(f"MTBF : {m:.0f} h")
    print(f"B10 : {duree_bp(10, beta, eta):.0f} h (10 % des appareils en panne)")
    for t in (1000, 2000, 4000):
        print(f"R({t} h) = {float(fiabilite(t, beta, eta)) * 100:.1f} %")
    print(f"Disponibilité (MTTR {a.mttr:g} h) : {disponibilite(m, a.mttr) * 100:.2f} %")
    opt = intervalle_optimal(beta, eta, a.cp, a.cc)
    if opt is None:
        print("Maintenance préventive par l'âge : pas rentable avec ces coûts.")
    else:
        T, cT, sans = opt
        print(f"Remplacement préventif conseillé toutes les {T:.0f} h : "
              f"{cT * 1000:.1f} € / 1 000 h contre {sans * 1000:.1f} € / 1 000 h sans préventif "
              f"({(1 - cT / sans) * 100:.0f} % d'économie)")


if __name__ == "__main__":
    main()
