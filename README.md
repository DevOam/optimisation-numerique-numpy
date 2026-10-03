# TP optimisation - version simple

Code du projet : https://github.com/DevOam/optimisation-numerique-numpy

J'ai séparé le travail en trois fichiers pour pouvoir comprendre chaque exercice :

- `src/regression.py` : exercice 1 ;
- `src/classification.py` : exercice 2 ;
- `src/clustering.py` : exercice 3 ;
- `src/optimizers.py` : les sept méthodes d'optimisation.

Les données sont dans `data`, les résultats dans `results` et les images dans
`figures`.

## Lancer une première fois

```bash
cd ~/Downloads/tp_optimisation_etudiant
python3 -m pip install -r requirements.txt
python3 run_all.py --profile quick
```

Le mode `quick` permet de vérifier le code sans attendre trop longtemps.

## Lancer la version complète

```bash
python3 run_all.py --profile full
```

Ce mode utilise les grandes partitions de MNIST et Fashion-MNIST. Il prend
beaucoup plus de temps.

## Ordre conseillé pour lire le code

1. Lire `src/optimizers.py`, surtout la ligne `theta = theta - eta * gradient`.
2. Lire `src/regression.py`, car c'est l'exercice le plus simple.
3. Lire `src/classification.py` pour la rétropropagation.
4. Lire `src/clustering.py` pour les centroïdes.

## Ce que j'ai exécuté

- les jeux CSV avec trois graines ;
- un essai court de MNIST et Fashion-MNIST avec une graine et deux époques ;
- les vérifications numériques des gradients ;
- les tableaux et les figures du rapport.
