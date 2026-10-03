"""Exercice 2 : réseau dense à trois couches, rétropropagation NumPy."""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from optimizers import METHODS, optimize
from utils import DATA, RESULTS, FIGURES, fit_standardizer, standardize, classification_metrics, save_rows, save_json, central_difference_check

ETA = {"gd": 0.08, "gd_search": 1.0, "sgd": 0.02, "momentum": 0.01,
       "adagrad": 0.04, "rmsprop": 0.002, "adam": 0.002}


class MLP:
    def __init__(self, dimensions, regularization=1e-4, seed=42):
        self.dimensions = dimensions
        self.regularization = regularization
        rng = np.random.default_rng(seed)
        self.shapes = []
        arrays = []
        for fan_in, fan_out in zip(dimensions[:-1], dimensions[1:]):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            W = rng.uniform(-limit, limit, size=(fan_in, fan_out))
            b = np.zeros(fan_out)
            self.shapes.extend([W.shape, b.shape]); arrays.extend([W.ravel(), b])
        self.theta0 = np.concatenate(arrays)

    def unpack(self, theta):
        arrays, offset = [], 0
        for shape in self.shapes:
            size = int(np.prod(shape)); arrays.append(theta[offset:offset+size].reshape(shape)); offset += size
        return arrays

    def forward(self, X, theta):
        W1, b1, W2, b2, W3, b3 = self.unpack(theta)
        H1 = np.tanh(X @ W1 + b1)
        H2 = np.tanh(H1 @ W2 + b2)
        Z = H2 @ W3 + b3
        Z -= Z.max(axis=1, keepdims=True)
        exp = np.exp(Z); P = exp / exp.sum(axis=1, keepdims=True)
        return H1, H2, P

    def loss_gradient(self, X, y, theta, gradient=True):
        W1, b1, W2, b2, W3, b3 = self.unpack(theta)
        H1, H2, P = self.forward(X, theta)
        n = len(y); classes = self.dimensions[-1]
        loss = -np.mean(np.log(np.clip(P[np.arange(n), y], 1e-12, 1.0)))
        loss += self.regularization * (np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3)) / 2
        if not gradient: return float(loss)
        Y = np.eye(classes)[y]
        D3 = (P - Y) / n
        gW3 = H2.T @ D3 + self.regularization * W3; gb3 = D3.sum(axis=0)
        D2 = (D3 @ W3.T) * (1 - H2**2)
        gW2 = H1.T @ D2 + self.regularization * W2; gb2 = D2.sum(axis=0)
        D1 = (D2 @ W2.T) * (1 - H1**2)
        gW1 = X.T @ D1 + self.regularization * W1; gb1 = D1.sum(axis=0)
        return np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2, gW3.ravel(), gb3])


def load_csv(name):
    df = pd.read_csv(DATA / name); features = [c for c in df if c not in {"label", "split"}]
    X = df[features].to_numpy(float); y = df["label"].to_numpy(int); s = df["split"].to_numpy()
    mean, std = fit_standardizer(X[s == "train"])
    return {p: (standardize(X[s == p], mean, std), y[s == p]) for p in ["train", "val", "test"]}


def load_images(name, profile):
    z = np.load(DATA / f"{name}_50k_10k_10k.npz")
    sizes = (3000, 1000, 1000) if profile == "quick" else (50000, 10000, 10000)
    xtr = z["x_train"][:sizes[0]].astype(np.float32) / 255.0
    xv = z["x_val"][:sizes[1]].astype(np.float32) / 255.0
    xt = z["x_test"][:sizes[2]].astype(np.float32) / 255.0
    mean = xtr.mean(axis=0)
    return {"train": (xtr-mean, z["y_train"][:sizes[0]].astype(int)),
            "val": (xv-mean, z["y_val"][:sizes[1]].astype(int)),
            "test": (xt-mean, z["y_test"][:sizes[2]].astype(int))}


def run(profile="quick"):
    jobs = [
        ("classification_spiral3", load_csv("classification_spiral3.csv"), [2,32,16,3]),
        ("classification_iris", load_csv("classification_iris.csv"), [4,32,16,3]),
        ("mnist", load_images("mnist", profile), [784,128,64,10]),
        ("fashion_mnist", load_images("fashion_mnist", profile), [784,128,64,10]),
    ]
    rows, histories, details, checks, visuals = [], {}, {}, {}, {}
    for name, parts, dims in jobs:
        image_job = name in {"mnist", "fashion_mnist"}
        epochs = (2 if profile == "quick" else 8) if image_job else (80 if profile == "quick" else 200)
        seeds = [42] if (profile == "quick" and image_job) else [42,123,2024]
        model = MLP(dims, seed=42)
        Xtr,ytr = parts["train"]; Xv,yv = parts["val"]; Xt,yt = parts["test"]
        def loss(theta): return model.loss_gradient(Xtr,ytr,theta,False)
        def grad(theta, ids):
            xx,yy = (Xtr,ytr) if ids is None else (Xtr[ids],ytr[ids])
            return model.loss_gradient(xx,yy,theta,True)
        def val(theta): return model.loss_gradient(Xv,yv,theta,False)
        if not image_job:
            checks[name] = central_difference_check(loss, grad, model.theta0, coordinates=10)
        for method in METHODS:
            for seed in seeds:
                # La même initialisation est conservée pour toutes les méthodes d'une graine.
                initial = MLP(dims, seed=seed).theta0
                result = optimize(initial, loss, grad, val, method=method, eta=ETA[method], epochs=epochs,
                                  n_samples=len(ytr), batch_size=(1024 if image_job else 32), seed=seed)
                probs = model.forward(Xt, result.best_theta)[-1]
                metrics = classification_metrics(yt, probs)
                row = {"dataset":name,"method":method,"seed":seed,"profile":profile,"eta":ETA[method],
                       "epochs":epochs,"best_epoch":result.best_epoch,"validation_loss":val(result.best_theta),
                       "time_s":result.elapsed,"updates":result.updates,"gradient_calls":result.gradient_calls,
                       "loss_calls":result.loss_calls, **{k:v for k,v in metrics.items() if not isinstance(v,(list,dict))}}
                rows.append(row)
                key=f"{name}:{method}:{seed}"; histories[key]=result.history
                details[key]={"confusion":metrics["confusion"],"f1_per_class":metrics["f1_per_class"]}
                if name not in visuals or row["validation_loss"] < visuals[name]["validation_loss"]:
                    visuals[name]={"validation_loss":row["validation_loss"],"theta":result.best_theta.copy(),
                                   "parts":parts,"dims":dims,"method":method}
    save_rows(RESULTS/"classification_results.csv",rows)
    save_json(RESULTS/"classification_histories.json",histories)
    save_json(RESULTS/"classification_details.json",details)
    save_json(RESULTS/"classification_gradient_checks.json",checks)
    plot_results(rows,histories,details,visuals)
    return rows


def plot_results(rows,histories,details,visuals):
    for dataset in sorted({r["dataset"] for r in rows}):
        plt.figure(figsize=(8,5))
        for method in METHODS:
            key=f"{dataset}:{method}:42"; h=histories[key]
            plt.plot([x["epoch"] for x in h],[x["validation"] for x in h],label=method)
        plt.xlabel("Époque");plt.ylabel("Entropie croisée validation");plt.title(dataset);plt.legend(ncol=2)
        plt.tight_layout();plt.savefig(FIGURES/f"classification_{dataset}_courbes.png",dpi=150);plt.close()
        best=max([r for r in rows if r["dataset"]==dataset and r["seed"]==42],key=lambda r:r["accuracy"])
        cm=np.array(details[f"{dataset}:{best['method']}:42"]["confusion"])
        plt.figure(figsize=(5,4));plt.imshow(cm,cmap="Blues");plt.colorbar();
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]): plt.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=7)
        plt.xlabel("Prédit");plt.ylabel("Réel");plt.title(f"{dataset} - {best['method']}")
        plt.tight_layout();plt.savefig(FIGURES/f"classification_{dataset}_confusion.png",dpi=150);plt.close()

    # Frontière de décision sur les spirales standardisées.
    item=visuals["classification_spiral3"];model=MLP(item["dims"]);X,y=item["parts"]["test"]
    allx=np.vstack([item["parts"][p][0] for p in ["train","val","test"]])
    lo=allx.min(axis=0)-.4;hi=allx.max(axis=0)+.4
    gx,gy=np.meshgrid(np.linspace(lo[0],hi[0],180),np.linspace(lo[1],hi[1],180));grid=np.c_[gx.ravel(),gy.ravel()]
    zz=model.forward(grid,item["theta"])[-1].argmax(axis=1).reshape(gx.shape)
    plt.figure(figsize=(6,5));plt.contourf(gx,gy,zz,alpha=.3,cmap="tab10");plt.scatter(X[:,0],X[:,1],c=y,cmap="tab10",s=18)
    plt.xlabel("x1 standardisé");plt.ylabel("x2 standardisé");plt.title(f"Spirales - frontière {item['method']}")
    plt.tight_layout();plt.savefig(FIGURES/"classification_spiral_frontiere.png",dpi=150);plt.close()

    # Projection PCA d'Iris.
    from sklearn.decomposition import PCA
    item=visuals["classification_iris"];X,y=item["parts"]["test"];Xp=PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(6,5));plt.scatter(Xp[:,0],Xp[:,1],c=y,cmap="tab10",s=35)
    plt.xlabel("Composante 1");plt.ylabel("Composante 2");plt.title("Iris - projection PCA du test")
    plt.tight_layout();plt.savefig(FIGURES/"classification_iris_pca.png",dpi=150);plt.close()

    # Exemples d'images bien/mal classées avec confiance.
    for name in ["mnist","fashion_mnist"]:
        item=visuals[name];model=MLP(item["dims"]);X,y=item["parts"]["test"];P=model.forward(X,item["theta"])[-1];pred=P.argmax(axis=1)
        good=np.where(pred==y)[0][:5];bad=np.where(pred!=y)[0][:5];ids=np.r_[good,bad]
        fig,axes=plt.subplots(2,5,figsize=(9,4))
        for ax,index in zip(axes.ravel(),ids):
            ax.imshow(X[index].reshape(28,28),cmap="gray");ax.set_title(f"r={y[index]} p={pred[index]}\n{P[index,pred[index]]:.2f}",fontsize=8);ax.axis("off")
        fig.suptitle(f"{name} : ligne 1 correcte, ligne 2 incorrecte");fig.tight_layout()
        fig.savefig(FIGURES/f"classification_{name}_exemples.png",dpi=150);plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--profile",choices=["quick","full"],default="quick");run(p.parse_args().profile)
