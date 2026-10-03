"""Exercice 1 : OLS, Ridge et Ridge à noyau RBF."""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from optimizers import METHODS, optimize
from utils import DATA, RESULTS, FIGURES, fit_standardizer, standardize, regression_metrics, save_rows, save_json, central_difference_check


ETA = {
    "gd": 0.05, "gd_search": 1.0, "sgd": 0.01, "momentum": 0.005,
    "adagrad": 0.05, "rmsprop": 0.005, "adam": 0.01,
}


def load_dataset(name):
    df = pd.read_csv(DATA / name)
    features = [c for c in df.columns if c not in {"y", "split"}]
    X = df[features].to_numpy(float); y = df["y"].to_numpy(float)
    masks = {part: (df["split"].to_numpy() == part) for part in ["train", "val", "test"]}
    Xm, Xs = fit_standardizer(X[masks["train"]])
    ym, ys = y[masks["train"]].mean(), y[masks["train"]].std()
    parts = {}
    for part, mask in masks.items():
        parts[part] = (standardize(X[mask], Xm, Xs), (y[mask] - ym) / ys, y[mask])
    return parts, ym, ys


def linear_problem(parts, ridge_lambda):
    Xtr = np.column_stack([np.ones(len(parts["train"][0])), parts["train"][0]])
    Xv = np.column_stack([np.ones(len(parts["val"][0])), parts["val"][0]])
    Xt = np.column_stack([np.ones(len(parts["test"][0])), parts["test"][0]])
    ytr, yv = parts["train"][1], parts["val"][1]
    penalty = np.eye(Xtr.shape[1]); penalty[0, 0] = 0.0
    def loss(w):
        return np.mean((Xtr @ w - ytr) ** 2) / 2 + ridge_lambda * (w @ penalty @ w) / 2
    def grad(w, ids):
        xx, yy = (Xtr, ytr) if ids is None else (Xtr[ids], ytr[ids])
        return xx.T @ (xx @ w - yy) / len(yy) + ridge_lambda * penalty @ w
    def val(w): return np.sqrt(np.mean((Xv @ w - yv) ** 2))
    H = Xtr.T @ Xtr / len(Xtr) + ridge_lambda * penalty
    def qstep(g):
        den = g @ H @ g
        return float((g @ g) / den) if den > 1e-14 else 0.0
    return np.zeros(Xtr.shape[1]), loss, grad, val, qstep, Xt


def rbf(A, B, sigma):
    distances = np.sum(A*A, axis=1)[:, None] + np.sum(B*B, axis=1)[None, :] - 2 * A @ B.T
    return np.exp(-np.maximum(distances, 0.0) / (2 * sigma**2))


def kernel_problem(parts, ridge_lambda, sigma):
    Xtr, ytr = parts["train"][0], parts["train"][1]
    K = rbf(Xtr, Xtr, sigma)
    Kv = rbf(parts["val"][0], Xtr, sigma)
    Kt = rbf(parts["test"][0], Xtr, sigma)
    def loss(a): return np.mean((K @ a - ytr) ** 2) / 2 + ridge_lambda * (a @ K @ a) / 2
    def grad(a, ids):
        if ids is None:
            return K @ (K @ a - ytr) / len(ytr) + ridge_lambda * K @ a
        KB = K[ids]
        return KB.T @ (KB @ a - ytr[ids]) / len(ids) + ridge_lambda * K @ a
    def val(a): return np.sqrt(np.mean((Kv @ a - parts["val"][1]) ** 2))
    H = K @ K / len(K) + ridge_lambda * K
    def qstep(g):
        den = g @ H @ g
        return float((g @ g) / den) if den > 1e-14 else 0.0
    return np.zeros(len(K)), loss, grad, val, qstep, Kt


def run(profile="quick"):
    datasets = ["regression_nonlinear.csv", "regression_diabetes.csv"]
    seeds = [42, 123, 2024]
    epochs = 120 if profile == "quick" else 400
    rows, histories, checks, visuals = [], {}, {}, {}
    for dataset in datasets:
        parts, ym, ys = load_dataset(dataset)
        configurations = [("ols", 0.0, None), ("ridge", 0.01, None), ("rbf", 0.01, 1.0)]
        for model, lam, sigma in configurations:
            problem = kernel_problem(parts, lam, sigma) if model == "rbf" else linear_problem(parts, lam)
            theta0, loss, grad, val, qstep, Xtest = problem
            checks[f"{dataset}:{model}"] = central_difference_check(loss, grad, theta0 + 0.01)
            for method in METHODS:
                run_seeds = seeds if method in {"sgd", "momentum", "adagrad", "rmsprop", "adam"} else [42]
                for seed in run_seeds:
                    result = optimize(theta0, loss, grad, val, method=method, eta=ETA[method], epochs=epochs,
                                      n_samples=len(parts["train"][1]), batch_size=32, seed=seed,
                                      quadratic_step_fn=qstep if method == "gd_search" else None)
                    pred_standard = Xtest @ result.best_theta
                    pred = pred_standard * ys + ym
                    metrics = regression_metrics(parts["test"][2], pred)
                    row = {"dataset": dataset, "model": model, "method": method, "seed": seed,
                           "eta": ETA[method], "lambda": lam, "sigma": sigma or "",
                           "best_epoch": result.best_epoch, "rmse_validation": val(result.best_theta) * ys,
                           "time_s": result.elapsed, "updates": result.updates,
                           "gradient_calls": result.gradient_calls, "loss_calls": result.loss_calls, **metrics}
                    rows.append(row)
                    histories[f"{dataset}:{model}:{method}:{seed}"] = result.history
                    if dataset not in visuals or row["rmse_validation"] < visuals[dataset]["rmse_validation"]:
                        visuals[dataset] = {"rmse_validation": row["rmse_validation"], "y": parts["test"][2],
                                            "pred": pred, "X": parts["test"][0], "label": f"{model} - {method}"}
    save_rows(RESULTS / "regression_results.csv", rows)
    save_json(RESULTS / "regression_histories.json", histories)
    save_json(RESULTS / "regression_gradient_checks.json", checks)
    plot_results(rows, histories, visuals)
    return rows


def plot_results(rows, histories, visuals):
    for dataset in sorted({r["dataset"] for r in rows}):
        plt.figure(figsize=(8, 5))
        for method in METHODS:
            key = f"{dataset}:ols:{method}:42"
            h = histories[key]
            plt.plot([x["epoch"] for x in h], [x["validation"] for x in h], label=method)
        plt.xlabel("Époque"); plt.ylabel("RMSE validation standardisée")
        plt.title(f"Régression OLS - {dataset}"); plt.legend(ncol=2); plt.tight_layout()
        plt.savefig(FIGURES / f"regression_{dataset[:-4]}_courbes.png", dpi=150); plt.close()

    # Comparaison synthétique des RMSE test, moyenne par méthode/modèle.
    labels, values = [], []
    selected = [r for r in rows if r["dataset"] == "regression_nonlinear.csv"]
    for model in ["ols", "ridge", "rbf"]:
        for method in METHODS:
            vals = [r["rmse"] for r in selected if r["model"] == model and r["method"] == method]
            labels.append(f"{model}\n{method}"); values.append(np.mean(vals))
    plt.figure(figsize=(12, 5)); plt.bar(np.arange(len(values)), values)
    plt.xticks(np.arange(len(values)), labels, rotation=70, fontsize=7)
    plt.ylabel("RMSE test"); plt.title("Régression non linéaire : comparaison des modèles")
    plt.tight_layout(); plt.savefig(FIGURES / "regression_comparaison_rmse.png", dpi=150); plt.close()

    for dataset, item in visuals.items():
        y, pred = item["y"], item["pred"]
        plt.figure(figsize=(5,5)); plt.scatter(y,pred,alpha=.75)
        low,high=min(y.min(),pred.min()),max(y.max(),pred.max());plt.plot([low,high],[low,high],"k--")
        plt.xlabel("Valeur réelle");plt.ylabel("Prédiction");plt.title(f"{dataset} - {item['label']}")
        plt.tight_layout();plt.savefig(FIGURES/f"regression_{dataset[:-4]}_predictions.png",dpi=150);plt.close()
        plt.figure(figsize=(6,4));plt.hist(pred-y,bins=15,edgecolor="black")
        plt.xlabel("Résidu (prédiction - réel)");plt.ylabel("Effectif");plt.title(f"Résidus test - {dataset}")
        plt.tight_layout();plt.savefig(FIGURES/f"regression_{dataset[:-4]}_residus.png",dpi=150);plt.close()
    corr=pd.read_csv(DATA/"regression_diabetes.csv").drop(columns=["split"]).corr()
    plt.figure(figsize=(7,6));plt.imshow(corr,cmap="coolwarm",vmin=-1,vmax=1);plt.colorbar();
    plt.xticks(range(len(corr)),corr.columns,rotation=75,fontsize=7);plt.yticks(range(len(corr)),corr.columns,fontsize=7)
    plt.title("Diabetes - corrélations");plt.tight_layout();plt.savefig(FIGURES/"regression_diabetes_correlations.png",dpi=150);plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    run(parser.parse_args().profile)
