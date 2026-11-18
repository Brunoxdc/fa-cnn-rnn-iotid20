"""Carga, limpieza, partición y auditoría de IoTID20."""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from . import config as C


def find_csv(path: str | None = None) -> str:
    """Devuelve la ruta del CSV de IoTID20.

    - Si `path` es un .csv, lo usa.
    - Si es una carpeta, busca el primer .csv dentro.
    - Si es None, lo descarga de Kaggle con kagglehub (no requiere login).
    """
    if path is None:
        import kagglehub  # se instala en Colab con pip
        path = kagglehub.dataset_download(C.KAGGLE_DATASET)
    if os.path.isfile(path):
        return path
    csvs = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
    if not csvs:
        raise FileNotFoundError(f"No se encontró ningún .csv en {path}")
    return max(csvs, key=os.path.getsize)  # el archivo grande es el dataset


def load_raw(path: str | None = None) -> pd.DataFrame:
    csv = find_csv(path)
    print(f"[data] leyendo {csv}")
    df = pd.read_csv(csv, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Elimina identificadores, convierte a numérico y repara inf/NaN.

    Devuelve X (features numéricas), y_bin (1 = ataque) y y_cat (categoría).
    """
    drop = [c for c in C.ID_COLUMNS + C.LABEL_COLUMNS if c in df.columns]
    X = df.drop(columns=drop).apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    # columnas constantes no aportan información
    const = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    X = X.drop(columns=const).astype("float32")
    y_bin = (df["Label"].astype(str).str.strip() == C.POSITIVE_LABEL).astype(int)
    y_cat = df["Cat"].astype(str).str.strip() if "Cat" in df.columns else y_bin.astype(str)
    print(f"[data] {X.shape[0]:,} filas x {X.shape[1]} features "
          f"({len(const)} constantes eliminadas)")
    return X, y_bin, y_cat


def audit(X: pd.DataFrame, y: pd.Series) -> dict:
    """Mide desbalance y duplicados exactos (posible fuga train/test)."""
    full = X.copy()
    full["_y"] = y.values
    n = len(full)
    n_unique = len(full.drop_duplicates())
    counts = y.value_counts().to_dict()

    # ¿Cuántas filas del test aparecen idénticas en train con una partición aleatoria?
    idx_tr, idx_te = train_test_split(
        np.arange(n), test_size=C.TEST_SIZE, stratify=y, random_state=C.SEED)
    h = pd.util.hash_pandas_object(full, index=False).values
    train_hashes = set(h[idx_tr])
    leak = float(np.mean([v in train_hashes for v in h[idx_te]]))

    rep = {
        "filas": n,
        "filas_unicas": n_unique,
        "duplicados": n - n_unique,
        "pct_duplicados": (n - n_unique) / n,
        "normal": int(counts.get(0, 0)),
        "ataque": int(counts.get(1, 0)),
        "pct_test_visto_en_train": leak,
    }
    print("[audit]", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep.items()})
    return rep


def split_scale(X: pd.DataFrame, y: pd.Series, dedup: bool = False):
    """Partición estratificada 70/30 + MinMax ajustado solo en train."""
    if dedup:
        full = X.copy()
        full["_y"] = y.values
        full = full.drop_duplicates()
        y = full.pop("_y")
        X = full
        print(f"[data] sin duplicados: {len(X):,} filas")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.SEED)
    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_tr).astype("float32")
    X_te = np.clip(scaler.transform(X_te), 0, 1).astype("float32")
    return X_tr, X_te, y_tr.to_numpy(), y_te.to_numpy()


def apply_smote(X_tr, y_tr):
    """SMOTE solo sobre train (el test queda con la distribución real)."""
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(k_neighbors=C.SMOTE_K, random_state=C.SEED)
    X_res, y_res = sm.fit_resample(X_tr, y_tr)
    print(f"[smote] train {len(y_tr):,} -> {len(y_res):,} "
          f"(clases: {np.bincount(y_res).tolist()})")
    return X_res.astype("float32"), y_res
