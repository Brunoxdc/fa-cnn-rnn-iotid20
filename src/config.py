"""Hiperparámetros de la réplica.

Los valores marcados con [PAPER] salen de Zayed, Alshathri & El-Shafai (2026),
Int. J. Comput. Intell. Syst. 19:158, https://doi.org/10.1007/s44196-026-01178-2

Los marcados con [SUPUESTO] no están especificados en el paper y son decisiones
de esta réplica (ver README, sección "Supuestos").
"""

SEED = 42

# --- Datos -------------------------------------------------------------------
KAGGLE_DATASET = "rohulaminlabid/iotid20-dataset"   # copia pública de IoTID20
TEST_SIZE = 0.30                                    # [PAPER] 70/30
# Columnas identificadoras: se eliminan para que el modelo no "memorice" IPs o
# timestamps (fuga de información). [SUPUESTO] el paper no lista columnas eliminadas.
ID_COLUMNS = ["Flow_ID", "Src_IP", "Dst_IP", "Timestamp"]
LABEL_COLUMNS = ["Label", "Cat", "Sub_Cat"]
POSITIVE_LABEL = "Anomaly"                          # ataque = clase positiva

# --- SMOTE --------------------------------------------------------------------
SMOTE_K = 5                 # [SUPUESTO] el paper no indica k (5 es el default)

# --- Firefly Algorithm (selección de características) ------------------------
FA_POP = 20                 # [PAPER] 20 luciérnagas
FA_ITERS = 50               # [PAPER] 50 iteraciones máx.
FA_ALPHA = 0.5              # [PAPER] alfa en [0,1] -> usamos 0.5 con decaimiento
FA_ALPHA_DECAY = 0.97       # [SUPUESTO]
FA_BETA0 = 0.1              # [PAPER] beta = 0.1 (el paper lo describe de forma inconsistente)
FA_GAMMA = 0.01             # [PAPER] gamma = 0.01
FA_FEATURE_PENALTY = 0.01   # [SUPUESTO] fitness = 0.99*acc + 0.01*(1 - k/n)
FA_SAMPLE = 10_000          # [SUPUESTO] filas usadas para evaluar el fitness (velocidad)
FA_TREE_DEPTH = 10          # [SUPUESTO] el paper no dice qué clasificador usa el fitness; usamos un árbol de decisión

# --- CNN-RNN -----------------------------------------------------------------
CONV_FILTERS = 64           # [PAPER]
CONV_KERNEL = 3             # [PAPER]
POOL_SIZE = 2               # [PAPER]
RNN_UNITS = 50              # [PAPER]
DROPOUT = 0.2               # [PAPER]
RNN_TYPE = "SimpleRNN"      # [SUPUESTO] el paper no dice LSTM/GRU; menciona GRU solo como trabajo futuro
LEARNING_RATE = 1e-3        # [PAPER] Adam
BATCH_SIZE = 64             # [PAPER]
EPOCHS = 10                 # [PAPER]

# --- Resultados reportados por el paper (IoTID20, binario) -------------------
PAPER_RESULTS_IOTID20 = {
    "FA-CNN-RNN": 0.982,
    "CNN-RNN (sin FA)": 0.942,
    "CNN": 0.92,
    "RNN": 0.86,
    "FA-CNN": 0.976,
    "FA-RNN": 0.93,
}
