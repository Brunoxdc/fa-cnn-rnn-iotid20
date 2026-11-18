"""Inserta la tabla de results/metrics.csv y la auditoría en el README.

Reemplaza lo que haya entre <!-- RESULTADOS:INICIO --> y <!-- RESULTADOS:FIN -->.
"""
import json
import re

import pandas as pd

df = pd.read_csv("results/metrics.csv")
s = json.load(open("results/summary.json", encoding="utf-8"))
a, fa = s["auditoria"], s["firefly"]

cols = ["experimento", "modelo", "particion", "n_features", "accuracy", "precision",
        "recall", "f1", "f1_macro", "recall_normal", "mcc"]
t = df[cols].copy()
for c in cols[4:]:
    t[c] = t[c].map(lambda v: f"{v:.4f}")
header = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
body = "".join("| " + " | ".join(map(str, r)) + " |\n" for r in t.to_numpy())

block = f"""<!-- RESULTADOS:INICIO -->
**Auditoría de IoTID20**

| Filas | Únicas | Duplicados | Normal | Ataque | Test visto en train (partición aleatoria) |
|---|---|---|---|---|---|
| {a['filas']:,} | {a['filas_unicas']:,} | {a['pct_duplicados']:.1%} | {a['normal']:,} | {a['ataque']:,} | {a['pct_test_visto_en_train']:.1%} |

**Firefly:** eligió {fa['n_seleccionadas']} de {fa['n_total']} columnas (fitness {fa['mejor_fitness']:.4f}, {fa['segundos']:.0f} s).

**Métricas en el conjunto de prueba** (clase positiva = ataque; batch {s['batch_size']}, {s['epocas']} épocas, {s['rnn']})

{header}{body}
Paper (IoTID20, binario): FA-CNN-RNN 98.2% · CNN-RNN 94.2% de accuracy.

![Comparación](figures/comparacion_paper_vs_replica.png)
<!-- RESULTADOS:FIN -->"""

readme = open("README.md", encoding="utf-8").read()
readme = re.sub(r"<!-- RESULTADOS:INICIO -->.*<!-- RESULTADOS:FIN -->", block, readme, flags=re.S)
open("README.md", "w", encoding="utf-8").write(readme)
print("README actualizado")
