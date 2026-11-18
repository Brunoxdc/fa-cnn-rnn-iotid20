"""Réplica de FA-CNN-RNN (Zayed et al., 2026) sobre IoTID20 - clasificación binaria.

Experimentos:
  E0  Auditoría del dataset (desbalance, duplicados, fuga train/test)
  E1  Réplica fiel: partición aleatoria 70/30 + SMOTE + Firefly + CNN-RNN
  E2  Ablación: CNN-RNN con todas las features (sin Firefly)
  E3  Baseline: Random Forest con las features del Firefly
  E4  Control de fuga: E1 y E3 repetidos tras eliminar filas duplicadas

Uso:
  python run_experiments.py                 # descarga IoTID20 de Kaggle (kagglehub)
  python run_experiments.py --data ruta.csv # usa un CSV local
  python run_experiments.py --fast          # batch 256 (más rápido, menos fiel)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from src import config as C
from src import data as D
from src import evaluate as E

RES, FIG = "results", "figures"


def train_cnn_rnn(X_tr, y_tr, X_te, y_te, name, batch):
    import tensorflow as tf
    from src.model import build_cnn_rnn
    tf.keras.utils.set_random_seed(C.SEED)
    model = build_cnn_rnn(X_tr.shape[1])
    t0 = time.time()
    hist = model.fit(X_tr[..., None], y_tr, epochs=C.EPOCHS, batch_size=batch, verbose=2)
    train_s = time.time() - t0
    prob = model.predict(X_te[..., None], batch_size=4096, verbose=0).ravel()
    y_pred = (prob >= 0.5).astype(int)
    return y_pred, train_s, hist.history, model.count_params()


def train_rf(X_tr, y_tr, X_te):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=C.SEED)
    t0 = time.time()
    rf.fit(X_tr, y_tr)
    return rf.predict(X_te), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="CSV o carpeta de IoTID20 (si no, Kaggle)")
    ap.add_argument("--fast", action="store_true", help="batch 256 en vez de 64")
    ap.add_argument("--max-rows", type=int, default=None, help="submuestra para pruebas rápidas")
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()
    batch = 256 if args.fast else C.BATCH_SIZE
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    raw = D.load_raw(args.data)
    if args.max_rows:
        raw = raw.sample(n=min(args.max_rows, len(raw)), random_state=C.SEED)
    X, y, _ = D.clean(raw)
    feat_names = np.array(X.columns)

    # ---------------- E0: auditoría ----------------
    audit = D.audit(X, y)
    rows = []

    def log(exp, model, split, n_feat, m, secs):
        r = {"experimento": exp, "modelo": model, "particion": split,
             "n_features": int(n_feat), "segundos_entreno": round(secs, 1), **m}
        rows.append(r)
        print(f"[{exp}] {model:<22} acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
              f"f1_macro={m['f1_macro']:.4f} recall_normal={m['recall_normal']:.4f} "
              f"mcc={m['mcc']:.4f}")
        pd.DataFrame(rows).to_csv(f"{RES}/metrics.csv", index=False)

    # ---------------- E1: réplica fiel ----------------
    X_tr, X_te, y_tr, y_te = D.split_scale(X, y, dedup=False)
    X_sm, y_sm = D.apply_smote(X_tr, y_tr)

    from src.firefly import BinaryFirefly
    t0 = time.time()
    fa = BinaryFirefly(X_sm, y_sm)
    mask, best_fit = fa.run()
    fa_s = time.time() - t0
    selected = feat_names[mask].tolist()
    print(f"[firefly] {len(selected)} features seleccionadas en {fa_s:.0f}s")
    E.plot_convergence(fa.history, f"{FIG}/firefly_convergencia.png")

    y_pred, secs, hist, n_params = train_cnn_rnn(X_sm[:, mask], y_sm, X_te[:, mask], y_te,
                                                 "FA-CNN-RNN", batch)
    m_e1 = E.metrics(y_te, y_pred)
    log("E1", "FA-CNN-RNN", "aleatoria", mask.sum(), m_e1, secs)
    E.plot_confusion(y_te, y_pred, "FA-CNN-RNN (partición aleatoria)", f"{FIG}/cm_fa_cnn_rnn.png")
    E.plot_history(hist, f"{FIG}/entrenamiento_fa_cnn_rnn.png")

    # ---------------- E2: ablación sin Firefly ----------------
    m_e2 = None
    if not args.skip_ablation:
        y_pred, secs, _, _ = train_cnn_rnn(X_sm, y_sm, X_te, y_te, "CNN-RNN", batch)
        m_e2 = E.metrics(y_te, y_pred)
        log("E2", "CNN-RNN (sin FA)", "aleatoria", X_sm.shape[1], m_e2, secs)

    # ---------------- E3: baseline Random Forest ----------------
    y_pred, secs = train_rf(X_sm[:, mask], y_sm, X_te[:, mask])
    m_e3 = E.metrics(y_te, y_pred)
    log("E3", "Random Forest (FA)", "aleatoria", mask.sum(), m_e3, secs)
    E.plot_confusion(y_te, y_pred, "Random Forest (partición aleatoria)", f"{FIG}/cm_rf.png")

    # ---------------- E4: sin duplicados ----------------
    Xd_tr, Xd_te, yd_tr, yd_te = D.split_scale(X, y, dedup=True)
    Xd_sm, yd_sm = D.apply_smote(Xd_tr, yd_tr)
    y_pred, secs, _, _ = train_cnn_rnn(Xd_sm[:, mask], yd_sm, Xd_te[:, mask], yd_te,
                                       "FA-CNN-RNN dedup", batch)
    m_e4 = E.metrics(yd_te, y_pred)
    log("E4", "FA-CNN-RNN", "sin duplicados", mask.sum(), m_e4, secs)
    E.plot_confusion(yd_te, y_pred, "FA-CNN-RNN (sin duplicados)", f"{FIG}/cm_fa_cnn_rnn_dedup.png")
    y_pred, secs = train_rf(Xd_sm[:, mask], yd_sm, Xd_te[:, mask])
    m_e4rf = E.metrics(yd_te, y_pred)
    log("E4", "Random Forest (FA)", "sin duplicados", mask.sum(), m_e4rf, secs)

    # ---------------- Resumen ----------------
    comp = [("FA-CNN-RNN", C.PAPER_RESULTS_IOTID20["FA-CNN-RNN"], m_e1["accuracy"])]
    if m_e2:
        comp.append(("CNN-RNN\n(sin FA)", C.PAPER_RESULTS_IOTID20["CNN-RNN (sin FA)"], m_e2["accuracy"]))
    comp += [("Random Forest\n(FA)", None, m_e3["accuracy"]),
             ("FA-CNN-RNN\nsin duplicados", None, m_e4["accuracy"])]
    E.plot_comparison(comp, f"{FIG}/comparacion_paper_vs_replica.png")

    summary = {
        "auditoria": audit,
        "firefly": {"features_seleccionadas": selected, "n_seleccionadas": len(selected),
                    "n_total": int(len(feat_names)), "mejor_fitness": best_fit,
                    "segundos": round(fa_s, 1)},
        "cnn_rnn_parametros": int(n_params),
        "batch_size": batch, "epocas": C.EPOCHS, "rnn": C.RNN_TYPE,
        "paper_iotid20_binario": C.PAPER_RESULTS_IOTID20,
    }
    with open(f"{RES}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    print("\nListo. Resultados en results/ y gráficas en figures/")
    print(pd.DataFrame(rows)[["experimento", "modelo", "particion", "accuracy", "f1",
                              "f1_macro", "recall_normal", "mcc"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
