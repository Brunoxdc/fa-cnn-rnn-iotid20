"""Firefly Algorithm binario para selección de características.

Cada luciérnaga es un vector continuo; una función sigmoide lo convierte en una
máscara 0/1 (qué columnas se usan). El "brillo" es el fitness:

    fitness = (1 - P) * accuracy + P * (1 - k / n)

donde k = nº de columnas elegidas. Las luciérnagas se mueven hacia las más
brillantes: x_i += beta0 * exp(-gamma * r^2) * (x_j - x_i) + alpha * (rand - 0.5)
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from . import config as C


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class BinaryFirefly:
    def __init__(self, X, y, seed: int = C.SEED):
        rng = np.random.default_rng(seed)
        n = min(C.FA_SAMPLE, len(y))
        idx = rng.choice(len(y), size=n, replace=False)
        Xs, ys = X[idx], y[idx]
        self.Xa, self.Xb, self.ya, self.yb = train_test_split(
            Xs, ys, test_size=0.3, stratify=ys, random_state=seed)
        self.n_feat = X.shape[1]
        self.rng = rng
        self.cache: dict[bytes, float] = {}
        self.history: list[float] = []

    def _mask(self, pos):
        m = _sigmoid(pos) > self.rng.random(pos.shape)
        if not m.any():
            m[self.rng.integers(self.n_feat)] = True
        return m

    def fitness(self, mask) -> float:
        key = np.packbits(mask).tobytes()
        if key not in self.cache:
            clf = DecisionTreeClassifier(max_depth=C.FA_TREE_DEPTH, random_state=C.SEED)
            clf.fit(self.Xa[:, mask], self.ya)
            acc = clf.score(self.Xb[:, mask], self.yb)
            p = C.FA_FEATURE_PENALTY
            self.cache[key] = (1 - p) * acc + p * (1 - mask.sum() / self.n_feat)
        return self.cache[key]

    def run(self, verbose: bool = True):
        pop = self.rng.normal(0, 1, size=(C.FA_POP, self.n_feat))
        masks = np.array([self._mask(p) for p in pop])
        fit = np.array([self.fitness(m) for m in masks])
        best_i = int(fit.argmax())
        best_mask, best_fit = masks[best_i].copy(), fit[best_i]
        alpha = C.FA_ALPHA
        for it in range(C.FA_ITERS):
            # cada luciérnaga se mueve hacia todas las más brillantes y luego se
            # evalúa una sola vez (20 evaluaciones por iteración, no 400)
            old_pop, old_fit = pop.copy(), fit.copy()
            for i in range(C.FA_POP):
                brighter = np.where(old_fit > old_fit[i])[0]
                for j in brighter:
                    r2 = np.sum((pop[i] - old_pop[j]) ** 2)
                    beta = C.FA_BETA0 * np.exp(-C.FA_GAMMA * r2)
                    pop[i] += beta * (old_pop[j] - pop[i])
                pop[i] += alpha * (self.rng.random(self.n_feat) - 0.5)
                masks[i] = self._mask(pop[i])
                fit[i] = self.fitness(masks[i])
                if fit[i] > best_fit:
                    best_fit, best_mask = fit[i], masks[i].copy()
            # la mejor luciérnaga hace un paseo aleatorio
            b = int(fit.argmax())
            pop[b] += alpha * (self.rng.random(self.n_feat) - 0.5)
            alpha *= C.FA_ALPHA_DECAY
            self.history.append(float(best_fit))
            if verbose and (it % 10 == 0 or it == C.FA_ITERS - 1):
                print(f"[firefly] iter {it + 1:02d}/{C.FA_ITERS}  mejor fitness={best_fit:.4f}  "
                      f"features={int(best_mask.sum())}/{self.n_feat}")
        return best_mask, best_fit
