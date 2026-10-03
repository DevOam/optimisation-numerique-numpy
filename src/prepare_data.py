"""Prépare les huit jeux de données du TP avec des partitions reproductibles."""

from pathlib import Path
import gzip
import struct
import urllib.request
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_iris, load_wine
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def fixed_splits(n, y=None, seed=2026):
    indices = np.arange(n)
    stratify = y if y is not None else None
    train, rest = train_test_split(indices, test_size=0.4, random_state=seed, stratify=stratify)
    rest_y = y[rest] if y is not None else None
    val, test = train_test_split(rest, test_size=0.5, random_state=seed, stratify=rest_y)
    split = np.empty(n, dtype=object)
    split[train], split[val], split[test] = "train", "val", "test"
    return split


def save_real_csvs():
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df["y"] = diabetes.target
    df["split"] = fixed_splits(len(df), seed=2026)
    df.to_csv(DATA / "regression_diabetes.csv", index=False)

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=[f"x{i+1}" for i in range(iris.data.shape[1])])
    df["label"] = iris.target
    df["split"] = fixed_splits(len(df), iris.target, seed=2026)
    df.to_csv(DATA / "classification_iris.csv", index=False)

    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=[f"x{i+1}" for i in range(wine.data.shape[1])])
    df["true_label"] = wine.target
    df.to_csv(DATA / "clustering_wine.csv", index=False)


URLS = {
    "mnist": "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "fashion_mnist": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
}


def download(url, destination):
    if not destination.exists():
        print("Téléchargement :", url)
        urllib.request.urlretrieve(url, destination)


def read_idx(path):
    with gzip.open(path, "rb") as stream:
        magic, size = struct.unpack(">II", stream.read(8))
        if magic == 2049:
            return np.frombuffer(stream.read(), dtype=np.uint8)
        rows, cols = struct.unpack(">II", stream.read(8))
        return np.frombuffer(stream.read(), dtype=np.uint8).reshape(size, rows * cols)


def save_image_npz(name):
    raw = DATA / "raw" / name
    raw.mkdir(parents=True, exist_ok=True)
    base = URLS[name]
    names = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    for filename in names:
        download(base + filename, raw / filename)
    x_all = read_idx(raw / names[0])
    y_all = read_idx(raw / names[1])
    x_test = read_idx(raw / names[2])
    y_test = read_idx(raw / names[3])
    rng = np.random.default_rng(2026)
    order = rng.permutation(len(x_all))
    train, val = order[:50000], order[50000:]
    np.savez_compressed(
        DATA / f"{name}_50k_10k_10k.npz",
        x_train=x_all[train], y_train=y_all[train],
        x_val=x_all[val], y_val=y_all[val],
        x_test=x_test, y_test=y_test,
    )


def main():
    DATA.mkdir(exist_ok=True)
    save_real_csvs()
    save_image_npz("mnist")
    save_image_npz("fashion_mnist")
    print("Les cinq jeux réels ont été préparés dans", DATA)


if __name__ == "__main__":
    main()

