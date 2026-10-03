"""Point d'entrée unique du TP."""

import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))

from regression import run as run_regression
from classification import run as run_classification
from clustering import run as run_clustering

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Exécuter les trois exercices")
    parser.add_argument("--profile",choices=["quick","full"],default="quick")
    parser.add_argument("--exercise",choices=["all","regression","classification","clustering"],default="all")
    args=parser.parse_args()
    if args.exercise in {"all","regression"}:run_regression(args.profile)
    if args.exercise in {"all","classification"}:run_classification(args.profile)
    if args.exercise in {"all","clustering"}:run_clustering(args.profile)
    print("Terminé. Résultats : results/ ; figures : figures/")

