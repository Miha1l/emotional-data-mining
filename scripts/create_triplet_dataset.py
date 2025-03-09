import pandas as pd

N_LABELS = 4
N_SAMPLES_PER_CLASS = 100
N_SAMPLES = N_SAMPLES_PER_CLASS * N_LABELS

df = pd.read_csv('../data/crowd_train_2000.csv')

anchors_df = df.groupby(['label']).head(N_SAMPLES_PER_CLASS).reset_index(drop=True)

triplets = []

for _, anchor in anchors_df.iterrows():
    pos_sample = df.loc[(df["label"] == anchor["label"]) & (df["id"] != anchor["id"])].sample(1)
    neg_sample = df.loc[(df["label"] != anchor["label"]) & (df["id"] != anchor["id"])].sample(1)

    triplets.append({
        "anchors": anchor["audio_path"],
        "positives": pos_sample["audio_path"].iloc[0],
        "negatives": neg_sample["audio_path"].iloc[0],
        "anchors_labels": anchor["label"],
        "positives_labels": pos_sample["label"].iloc[0],
        "negatives_labels": neg_sample["label"].iloc[0],
    })

triplet_df = pd.DataFrame(triplets)
triplet_df.to_csv(f"../data/triplets_{len(triplets)}.csv", index=False)
