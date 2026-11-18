"""Dibuja docs/workflow.png (diagrama del pipeline de la réplica)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ORANGE, DARK, GREY, LIGHT = "#F04E23", "#1F1F1F", "#6B6B6B", "#FDE7E0"

steps = [
    ("1. Datos", "IoTID20\n625,783 flujos\n(Kaggle, público)"),
    ("2. Limpieza", "Quitar IDs (IP,\ntimestamp), inf/NaN,\nMinMax"),
    ("3. Partición", "70% train\n30% test\n(estratificada)"),
    ("4. SMOTE", "Balancea solo\nel train\nNormal = Ataque"),
    ("5. Firefly", "20 luciérnagas\n50 iteraciones\nelige columnas"),
    ("6. CNN ‖ RNN", "Conv1D 64 + RNN 50\nconcat → sigmoide\nAdam, 10 épocas"),
    ("7. Evaluación", "Accuracy, F1,\nrecall Normal, MCC\nvs. paper"),
]

fig, ax = plt.subplots(figsize=(16, 4.6), dpi=200)
ax.set_xlim(0, 16)
ax.set_ylim(0, 4.6)
ax.axis("off")
w, h, gap, y = 1.95, 1.75, 0.27, 2.35
for i, (title, body) in enumerate(steps):
    x = 0.3 + i * (w + gap)
    hl = title.startswith(("4", "5", "6"))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=ORANGE if hl else "white", ec=ORANGE, lw=2))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=12.5,
            fontweight="bold", color="white" if hl else DARK)
    ax.text(x + w / 2, y + 0.68, body, ha="center", va="center", fontsize=9.6,
            color="white" if hl else GREY, linespacing=1.35)
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((x + w + 0.02, y + h / 2), (x + w + gap - 0.02, y + h / 2),
                                     arrowstyle="-|>", mutation_scale=14, color=DARK, lw=1.5))

ax.text(0.3 + 3 * (w + gap) + w / 2 + (w + gap), y + h + 0.32, "Núcleo del paper: FA-CNN-RNN",
        ha="center", fontsize=11, color=ORANGE, fontweight="bold")

# controles añadidos por la réplica
cy, ch = 0.25, 1.35
controls = [
    (0.3 + 1 * (w + gap), 2 * w + gap,
     "Control de fuga (aporte)", "Repetir todo sin filas duplicadas\npara ver si el test ya estaba en el train"),
    (0.3 + 4 * (w + gap), 2 * w + gap,
     "Ablación + baseline (aporte)", "CNN-RNN sin Firefly y Random Forest\ncon las mismas columnas"),
]
for x, cw, t, b in controls:
    ax.add_patch(FancyBboxPatch((x, cy), cw, ch, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=LIGHT, ec=ORANGE, lw=1.4, ls="--"))
    ax.text(x + cw / 2, cy + ch - 0.32, t, ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=DARK)
    ax.text(x + cw / 2, cy + 0.45, b, ha="center", va="center", fontsize=9.6, color=GREY)

fig.savefig("docs/workflow.png", bbox_inches="tight", facecolor="white")
print("docs/workflow.png")
