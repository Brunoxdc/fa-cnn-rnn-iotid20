"""Genera notebooks/replica_FA_CNN_RNN_IoTID20.ipynb."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# Réplica FA-CNN-RNN · IoTID20\n\n"
       "Réplica de **Zayed, Alshathri & El-Shafai (2026)**, *Firefly Algorithm Optimized Hybrid Deep "
       "Learning Framework for Intrusion Detection in IoT Environments*, IJCIS 19:158. "
       "[DOI](https://doi.org/10.1007/s44196-026-01178-2)\n\n"
       "**Antes de empezar:** Entorno de ejecución → Cambiar tipo de entorno → **GPU (T4)**.\n\n"
       "Tiempo aproximado con GPU: 35–45 min en modo fiel (batch 64) y 10–15 min con "
       "`FAST = True` (batch 256)."),
    code('REPO_URL = "https://github.com/Brunoxdc/fa-cnn-rnn-iotid20.git"\n'
         'FAST = False   # True = batch 256 (más rápido, menos fiel al paper)'),
    code('import os\n'
         'if not os.path.exists("fa-cnn-rnn-iotid20"):\n'
         '    !git clone -q {REPO_URL}\n'
         '%cd fa-cnn-rnn-iotid20\n'
         '!pip install -q imbalanced-learn kagglehub'),
    code('import tensorflow as tf\n'
         'print("TensorFlow", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU"))'),
    md("## 1. Descargar IoTID20 (Kaggle, público, sin login)"),
    code('import kagglehub\n'
         'DATA_DIR = kagglehub.dataset_download("rohulaminlabid/iotid20-dataset")\n'
         '!ls -lh "{DATA_DIR}"'),
    md("## 2. Ejecutar los experimentos E0–E4\n"
       "E0 auditoría · E1 réplica fiel · E2 sin Firefly · E3 Random Forest · E4 sin duplicados"),
    code('flag = "--fast" if FAST else ""\n'
         '!python -u run_experiments.py --data "{DATA_DIR}" {flag} 2>&1 | '
         'grep -v -E "I0000|W0000|E0000|oneDNN|cpu_feature|GrapplerHook|empty op name|does not specify"'),
    md("## 3. Resultados"),
    code('import json, pandas as pd\n'
         'from IPython.display import Image, display\n'
         's = json.load(open("results/summary.json"))\n'
         'print("Auditoría:", s["auditoria"])\n'
         'print("Firefly eligió", s["firefly"]["n_seleccionadas"], "de", s["firefly"]["n_total"], "columnas:")\n'
         'print(s["firefly"]["features_seleccionadas"])\n'
         'pd.read_csv("results/metrics.csv").round(4)'),
    code('for f in ["comparacion_paper_vs_replica", "cm_fa_cnn_rnn", "cm_fa_cnn_rnn_dedup", "cm_rf",\n'
         '          "firefly_convergencia", "entrenamiento_fa_cnn_rnn"]:\n'
         '    print(f); display(Image(f"figures/{f}.png", width=520))'),
    md("## 4. Guardar resultados\n"
       "Actualiza el README con la tabla y descarga un zip con `results/`, `figures/` y el README."),
    code('!python scripts/update_readme.py\n'
         '!zip -qr resultados_replica.zip results figures README.md\n'
         'from google.colab import files\n'
         'files.download("resultados_replica.zip")'),
    md("Para subir los resultados al repo, descomprime el zip en tu copia local y ejecuta:\n"
       "```\ngit add results figures README.md\n"
       'git commit -m "Resultados de la réplica en IoTID20"\ngit push\n```'),
]
nb.metadata = {"accelerator": "GPU", "colab": {"provenance": []},
               "kernelspec": {"name": "python3", "display_name": "Python 3"}}
nbf.write(nb, "notebooks/replica_FA_CNN_RNN_IoTID20.ipynb")
print("notebooks/replica_FA_CNN_RNN_IoTID20.ipynb")
