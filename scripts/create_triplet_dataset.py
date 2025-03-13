import pandas as pd

N_LABELS = 4
N_SAMPLES_PER_CLASS = 100
N_SAMPLES = N_SAMPLES_PER_CLASS * N_LABELS

df = pd.read_csv('../data/crowd_train_2000.csv')

anchors_df = df.groupby(['label']).head(N_SAMPLES_PER_CLASS).reset_index(drop=True)

triplets = []

for _, anchor in anchors_df.iterrows():
    pos_samples = df.loc[(df["label"] == anchor["label"]) & (df["id"] != anchor["id"])].sample(3)
    neg_samples = df.loc[df["label"] != anchor["label"]].groupby('label').sample(1)

    for i in range(3):
        triplets.append({
            "anchor": anchor["audio_path"],
            "positive": pos_samples["audio_path"].iloc[i],
            "negative": neg_samples["audio_path"].iloc[i],
            "anchor_label": anchor["label"],
            "positive_label": pos_samples["label"].iloc[i],
            "negative_label": neg_samples["label"].iloc[i],
        })

triplet_df = pd.DataFrame(triplets)
triplet_df.to_csv(f"../data/triplets_{len(triplets)}.csv", index=False)
