"""Exercice 3 : clustering doux par optimisation des centroïdes."""

import argparse
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.decomposition import PCA

from optimizers import METHODS, optimize
from utils import DATA, RESULTS, FIGURES, fit_standardizer, standardize, save_rows, save_json, central_difference_check

ETA = {"gd":0.05,"gd_search":1.0,"sgd":0.02,"momentum":0.01,"adagrad":0.08,"rmsprop":0.01,"adam":0.01}


def distances(X,C):
    return np.maximum(np.sum(X*X,axis=1)[:,None]+np.sum(C*C,axis=1)[None,:]-2*X@C.T,0.0)


def soft_loss_gradient(X,C,tau,gradient=True):
    D=distances(X,C); logits=-D/tau; maximum=logits.max(axis=1,keepdims=True)
    exp=np.exp(logits-maximum); Q=exp/exp.sum(axis=1,keepdims=True)
    log_mean=maximum[:,0]+np.log(exp.sum(axis=1))-np.log(C.shape[0])
    loss=-tau*np.mean(log_mean)
    if not gradient:return float(loss)
    return 2.0/len(X)*((Q.sum(axis=0)[:,None]*C)-Q.T@X)


def load_csv(name):
    df=pd.read_csv(DATA/name); label="true_label"; features=[c for c in df if c!=label]
    X=df[features].to_numpy(float); mean,std=fit_standardizer(X)
    return standardize(X,mean,std),df[label].to_numpy(int)


def load_images(name,profile):
    z=np.load(DATA/f"{name}_50k_10k_10k.npz")
    sizes=(3000,1000,1000) if profile=="quick" else (50000,10000,10000)
    return ((z["x_train"][:sizes[0]].astype(np.float32)/255,z["y_train"][:sizes[0]].astype(int)),
            (z["x_val"][:sizes[1]].astype(np.float32)/255,z["y_val"][:sizes[1]].astype(int)),
            (z["x_test"][:sizes[2]].astype(np.float32)/255,z["y_test"][:sizes[2]].astype(int)))


def external_metrics(X,y,C,seed=42):
    assignment=distances(X,C).argmin(axis=1); counts=np.bincount(assignment,minlength=len(C))
    inertia=float(np.mean(np.min(distances(X,C),axis=1)))
    rng=np.random.default_rng(seed); ids=rng.choice(len(X),size=min(1000,len(X)),replace=False)
    silhouette=float(silhouette_score(X[ids],assignment[ids])) if len(np.unique(assignment[ids]))>1 else float("nan")
    return {"soft_loss":None,"inertia":inertia,"non_empty":int(np.sum(counts>0)),
            "silhouette":silhouette,"ari":float(adjusted_rand_score(y,assignment)),
            "nmi":float(normalized_mutual_info_score(y,assignment)),"counts":counts.tolist()}


def lloyd(X,K,seed,iterations=50):
    rng=np.random.default_rng(seed); C=X[rng.choice(len(X),K,replace=False)].copy();start=perf_counter()
    for epoch in range(iterations):
        assignment=distances(X,C).argmin(axis=1);new=C.copy()
        for k in range(K):
            if np.any(assignment==k):new[k]=X[assignment==k].mean(axis=0)
        if np.allclose(new,C):break
        C=new
    return C,epoch+1,perf_counter()-start


def run(profile="quick"):
    bx,by=load_csv("clustering_blobs3.csv");wx,wy=load_csv("clustering_wine.csv")
    jobs=[("blobs",(bx,by),(bx,by),(bx,by),3,0.25),
          ("wine",(wx,wy),(wx,wy),(wx,wy),3,0.25)]
    for name in ["mnist","fashion_mnist"]:
        tr,va,te=load_images(name,profile);jobs.append((name,tr,va,te,10,5.0))
    rows,histories,details,checks={}, {}, {}, {}
    all_rows=[]
    for name,(Xtr,ytr),(Xv,yv),(Xt,yt),K,tau in jobs:
        image=name in {"mnist","fashion_mnist"};epochs=(2 if profile=="quick" else 8) if image else (60 if profile=="quick" else 200)
        seeds=[42] if (profile=="quick" and image) else [42,123,2024]
        for seed in seeds:
            rng=np.random.default_rng(seed);C0=Xtr[rng.choice(len(Xtr),K,replace=False)].copy();theta0=C0.ravel()
            def loss(theta):return soft_loss_gradient(Xtr,theta.reshape(K,-1),tau,False)
            def grad(theta,ids):
                xx=Xtr if ids is None else Xtr[ids]
                return soft_loss_gradient(xx,theta.reshape(K,-1),tau,True).ravel()
            def val(theta):return soft_loss_gradient(Xv,theta.reshape(K,-1),tau,False)
            if seed==42 and not image:checks[name]=central_difference_check(loss,grad,theta0,coordinates=10)
            for method in METHODS:
                result=optimize(theta0,loss,grad,val,method=method,eta=ETA[method],epochs=epochs,
                                n_samples=len(Xtr),batch_size=(2048 if image else 64),seed=seed)
                C=result.best_theta.reshape(K,-1);metrics=external_metrics(Xt,yt,C,seed);metrics["soft_loss"]=val(result.best_theta)
                row={"dataset":name,"method":method,"seed":seed,"profile":profile,"eta":ETA[method],"tau":tau,
                     "epochs":epochs,"best_epoch":result.best_epoch,"time_s":result.elapsed,"updates":result.updates,
                     "gradient_calls":result.gradient_calls,"loss_calls":result.loss_calls,
                     **{k:v for k,v in metrics.items() if not isinstance(v,list)}}
                all_rows.append(row);key=f"{name}:{method}:{seed}";histories[key]=result.history;details[key]={"counts":metrics["counts"],"centroids":C.tolist()}
            C,iterations,elapsed=lloyd(Xtr,K,seed)
            metrics=external_metrics(Xt,yt,C,seed);metrics["soft_loss"]=soft_loss_gradient(Xv,C,tau,False)
            all_rows.append({"dataset":name,"method":"lloyd","seed":seed,"profile":profile,"eta":"","tau":tau,
                             "epochs":iterations,"best_epoch":iterations,"time_s":elapsed,"updates":iterations,
                             "gradient_calls":0,"loss_calls":0,**{k:v for k,v in metrics.items() if not isinstance(v,list)}})
            details[f"{name}:lloyd:{seed}"]={"counts":metrics["counts"],"centroids":C.tolist()}
    save_rows(RESULTS/"clustering_results.csv",all_rows);save_json(RESULTS/"clustering_histories.json",histories)
    save_json(RESULTS/"clustering_details.json",details);save_json(RESULTS/"clustering_gradient_checks.json",checks)
    plot_results(all_rows,histories,details,jobs);return all_rows


def plot_results(rows,histories,details,jobs):
    for name,train,val,test,K,tau in jobs:
        plt.figure(figsize=(8,5))
        for method in METHODS:
            h=histories[f"{name}:{method}:42"]
            plt.plot([x["epoch"] for x in h],[x["validation"] for x in h],label=method)
        plt.xlabel("Époque");plt.ylabel("Perte douce validation");plt.title(f"Clustering - {name}");plt.legend(ncol=2)
        plt.tight_layout();plt.savefig(FIGURES/f"clustering_{name}_courbes.png",dpi=150);plt.close()
        X,y=test;best=min([r for r in rows if r["dataset"]==name and r["seed"]==42],key=lambda r:r["soft_loss"])
        C=np.array(details[f"{name}:{best['method']}:42"]["centroids"]);assignment=distances(X,C).argmin(axis=1)
        if X.shape[1]>2:
            pca=PCA(n_components=2,random_state=42);Xp=pca.fit_transform(X);Cp=pca.transform(C)
        else:Xp,Cp=X,C
        ids=np.arange(min(2000,len(X)))
        plt.figure(figsize=(6,5));plt.scatter(Xp[ids,0],Xp[ids,1],c=assignment[ids],s=8,cmap="tab10",alpha=.65)
        plt.scatter(Cp[:,0],Cp[:,1],c="black",marker="X",s=110,label="centroïdes");plt.legend();plt.title(f"{name} - {best['method']} (projection 2D)")
        plt.tight_layout();plt.savefig(FIGURES/f"clustering_{name}_projection.png",dpi=150);plt.close()
        if name in {"mnist","fashion_mnist"}:
            fig,axes=plt.subplots(2,5,figsize=(9,4));
            for k,ax in enumerate(axes.ravel()):ax.imshow(C[k].reshape(28,28),cmap="gray");ax.set_title(f"C{k}");ax.axis("off")
            fig.suptitle(f"Centroïdes reconstruits - {name}");fig.tight_layout();fig.savefig(FIGURES/f"clustering_{name}_centroides.png",dpi=150);plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--profile",choices=["quick","full"],default="quick");run(p.parse_args().profile)

