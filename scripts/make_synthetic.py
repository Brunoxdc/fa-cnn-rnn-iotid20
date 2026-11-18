"""Genera un CSV sintético con el MISMO esquema que IoTID20.

Sirve solo para probar que el pipeline corre de punta a punta sin descargar
los 308 MB reales. Los números que salen con este archivo NO son resultados.

    python scripts/make_synthetic.py data/synthetic_iotid20.csv 20000
"""
import sys

import numpy as np
import pandas as pd

FEATURES = ["Src_Port", "Dst_Port", "Protocol", "Flow_Duration", "Tot_Fwd_Pkts", "Tot_Bwd_Pkts",
            "TotLen_Fwd_Pkts", "TotLen_Bwd_Pkts", "Fwd_Pkt_Len_Max", "Fwd_Pkt_Len_Min",
            "Fwd_Pkt_Len_Mean", "Fwd_Pkt_Len_Std", "Bwd_Pkt_Len_Max", "Bwd_Pkt_Len_Min",
            "Bwd_Pkt_Len_Mean", "Bwd_Pkt_Len_Std", "Flow_Byts/s", "Flow_Pkts/s", "Flow_IAT_Mean",
            "Flow_IAT_Std", "Flow_IAT_Max", "Flow_IAT_Min", "Fwd_IAT_Tot", "Fwd_IAT_Mean",
            "Fwd_IAT_Std", "Fwd_IAT_Max", "Fwd_IAT_Min", "Bwd_IAT_Tot", "Bwd_IAT_Mean",
            "Bwd_IAT_Std", "Bwd_IAT_Max", "Bwd_IAT_Min", "Fwd_PSH_Flags", "Bwd_PSH_Flags",
            "Fwd_URG_Flags", "Bwd_URG_Flags", "Fwd_Header_Len", "Bwd_Header_Len", "Fwd_Pkts/s",
            "Bwd_Pkts/s", "Pkt_Len_Min", "Pkt_Len_Max", "Pkt_Len_Mean", "Pkt_Len_Std",
            "Pkt_Len_Var", "FIN_Flag_Cnt", "SYN_Flag_Cnt", "RST_Flag_Cnt", "PSH_Flag_Cnt",
            "ACK_Flag_Cnt", "URG_Flag_Cnt", "CWE_Flag_Count", "ECE_Flag_Cnt", "Down/Up_Ratio",
            "Pkt_Size_Avg", "Fwd_Seg_Size_Avg", "Bwd_Seg_Size_Avg", "Fwd_Byts/b_Avg",
            "Fwd_Pkts/b_Avg", "Fwd_Blk_Rate_Avg", "Bwd_Byts/b_Avg", "Bwd_Pkts/b_Avg",
            "Bwd_Blk_Rate_Avg", "Subflow_Fwd_Pkts", "Subflow_Fwd_Byts", "Subflow_Bwd_Pkts",
            "Subflow_Bwd_Byts", "Init_Fwd_Win_Byts", "Init_Bwd_Win_Byts", "Fwd_Act_Data_Pkts",
            "Fwd_Seg_Size_Min", "Active_Mean", "Active_Std", "Active_Max", "Active_Min",
            "Idle_Mean", "Idle_Std", "Idle_Max", "Idle_Min"]
CATS = ["Mirai", "DoS", "Scan", "MITM ARP Spoofing"]


def main(out, n):
    rng = np.random.default_rng(0)
    n_normal = int(n * 0.064)                     # ~6.4% normal, como IoTID20
    y = np.r_[np.zeros(n_normal, int), np.ones(n - n_normal, int)]
    X = rng.gamma(2.0, 50.0, size=(n, len(FEATURES)))
    X[y == 1, :10] *= rng.uniform(1.2, 2.0, size=(1, 10))  # señal débil en 10 columnas
    X[:, FEATURES.index("Fwd_PSH_Flags")] = 0      # columna constante
    X[rng.random(n) < 0.001, FEATURES.index("Flow_Byts/s")] = np.inf
    df = pd.DataFrame(X, columns=FEATURES)
    dup = rng.choice(n, size=n // 2)                # ~50% duplicados exactos
    df.iloc[n // 2:] = df.iloc[dup[: n - n // 2]].to_numpy()
    y[n // 2:] = y[dup[: n - n // 2]]
    df.insert(0, "Flow_ID", [f"f{i}" for i in range(n)])
    df.insert(1, "Src_IP", "192.168.0.13")
    df.insert(2, "Dst_IP", "192.168.0.16")
    df.insert(3, "Timestamp", "01/01/2019 00:00")
    df["Label"] = np.where(y == 1, "Anomaly", "Normal")
    df["Cat"] = np.where(y == 1, rng.choice(CATS, n), "Normal")
    df["Sub_Cat"] = df["Cat"]
    df.to_csv(out, index=False)
    print(f"escrito {out}: {n} filas")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20000)
