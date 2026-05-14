import pandas as pd
from src.config import DATA_DIR, RESULTS_DIR, SUBSAMPLED_FILE, CHUNK_SIZE, N_CHUNKS, SEQUENTIAL_TABLE_FILE


def main():

    df = pd.read_csv(DATA_DIR / SUBSAMPLED_FILE, parse_dates=["dteday"])
    chunks = [df.iloc[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE] for i in range(N_CHUNKS)]

    rows = []
    for i, c in enumerate(chunks):
        rows.append({
            "chunk":         i + 1,
            "date_start":    str(c["dteday"].min().date()),
            "date_end":      str(c["dteday"].max().date()),
            "cnt_mean":      round(c["cnt"].mean(), 1),
            "cnt_std":       round(c["cnt"].std(), 1),
            "temp_mean":     round(c["temp"].mean(), 3),
            "hum_mean":      round(c["hum"].mean(), 3),
            "corr_cnt_temp": round(c["cnt"].corr(c["temp"]), 3),
            "corr_cnt_hum":  round(c["cnt"].corr(c["hum"]), 3),
            "corr_temp_hum": round(c["temp"].corr(c["hum"]), 3),
        })

    diag = pd.DataFrame(rows)


    post = pd.read_csv(RESULTS_DIR / SEQUENTIAL_TABLE_FILE)
    beta2 = post[post["parameter"] == "beta_2_hum"][
        ["chunk", "mean", "ci_lower", "ci_upper"]
    ].rename(columns={
        "mean":     "beta2_mean",
        "ci_lower": "beta2_ci_lower",
        "ci_upper": "beta2_ci_upper",
    })

    out = diag.merge(beta2, on="chunk")
    out.to_csv(RESULTS_DIR / "chunk_diagnostics.csv", index=False)
    print(out.to_string(index=False))
    print("\nSaved to results/chunk_diagnostics.csv")


if __name__ == "__main__":
    main()