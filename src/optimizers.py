"""Les 7 méthodes d'optimisation demandées dans le TP."""

import time
import numpy as np

METHODS = ["gd", "gd_search", "sgd", "momentum", "adagrad", "rmsprop", "adam"]


class Resultat:
    """Petite classe pour ranger les résultats d'un entraînement."""
    pass


def recherche_pas(theta, gradient, fonction_perte):
    # Recherche simple entre 0 et 10.
    gauche, droite = 1e-8, 10.0
    nombre_or = (np.sqrt(5) - 1) / 2
    c = droite - nombre_or * (droite - gauche)
    d = gauche + nombre_or * (droite - gauche)
    fc = fonction_perte(theta - c * gradient)
    fd = fonction_perte(theta - d * gradient)
    appels = 2

    for _ in range(22):
        if fc < fd:
            droite, d, fd = d, c, fc
            c = droite - nombre_or * (droite - gauche)
            fc = fonction_perte(theta - c * gradient)
        else:
            gauche, c, fc = c, d, fd
            d = gauche + nombre_or * (droite - gauche)
            fd = fonction_perte(theta - d * gradient)
        appels += 1
    return (gauche + droite) / 2, appels


def optimize(theta0, loss_fn, grad_fn, validation_fn, method="gd", eta=0.01,
             epochs=100, n_samples=None, batch_size=None, seed=42,
             momentum=0.9, rho=0.9, beta1=0.9, beta2=0.999,
             eps=1e-8, quadratic_step_fn=None):
    """Boucle commune d'entraînement."""
    rng = np.random.default_rng(seed)
    theta = theta0.copy().astype(float)
    vitesse = np.zeros_like(theta)
    moyenne = np.zeros_like(theta)
    carres = np.zeros_like(theta)

    historique = []
    meilleur_theta = theta.copy()
    meilleure_validation = np.inf
    meilleure_epoque = 0
    mises_a_jour = 0
    appels_gradient = 0
    appels_perte = 0
    debut = time.perf_counter()
    methodes_mini_lots = ["sgd", "momentum", "adagrad", "rmsprop", "adam"]

    for epoque in range(1, epochs + 1):
        if method in methodes_mini_lots:
            ordre = rng.permutation(n_samples)
            lots = []
            for i in range(0, n_samples, batch_size):
                lots.append(ordre[i:i + batch_size])
        else:
            lots = [None]

        for indices in lots:
            gradient = grad_fn(theta, indices)
            appels_gradient += 1
            mises_a_jour += 1

            if method == "gd":
                theta = theta - eta * gradient
            elif method == "gd_search":
                if quadratic_step_fn is not None:
                    pas = quadratic_step_fn(gradient)
                else:
                    pas, nb = recherche_pas(theta, gradient, loss_fn)
                    appels_perte += nb
                theta = theta - pas * gradient
            elif method == "sgd":
                theta = theta - eta * gradient
            elif method == "momentum":
                vitesse = momentum * vitesse + gradient
                theta = theta - eta * vitesse
            elif method == "adagrad":
                carres = carres + gradient ** 2
                theta = theta - eta * gradient / (np.sqrt(carres) + eps)
            elif method == "rmsprop":
                carres = rho * carres + (1 - rho) * gradient ** 2
                theta = theta - eta * gradient / (np.sqrt(carres) + eps)
            elif method == "adam":
                moyenne = beta1 * moyenne + (1 - beta1) * gradient
                carres = beta2 * carres + (1 - beta2) * gradient ** 2
                moyenne_corrigee = moyenne / (1 - beta1 ** mises_a_jour)
                carres_corriges = carres / (1 - beta2 ** mises_a_jour)
                theta = theta - eta * moyenne_corrigee / (np.sqrt(carres_corriges) + eps)

        perte_train = float(loss_fn(theta))
        perte_validation = float(validation_fn(theta))
        appels_perte += 2
        historique.append({"epoch": epoque, "train_loss": perte_train,
                            "validation": perte_validation,
                            "time": time.perf_counter() - debut})

        if perte_validation < meilleure_validation:
            meilleure_validation = perte_validation
            meilleur_theta = theta.copy()
            meilleure_epoque = epoque

    resultat = Resultat()
    resultat.theta = theta
    resultat.best_theta = meilleur_theta
    resultat.best_epoch = meilleure_epoque
    resultat.history = historique
    resultat.updates = mises_a_jour
    resultat.gradient_calls = appels_gradient
    resultat.loss_calls = appels_perte
    resultat.elapsed = time.perf_counter() - debut
    return resultat

