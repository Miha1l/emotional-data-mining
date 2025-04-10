import pandas as pd

N_LABELS = 4
N_SAMPLES_PER_CLASS = 15
# N_SAMPLES = N_SAMPLES_PER_CLASS * N_LABELS


def create_triplet_dataset(filepath, n_labels, n_samples_per_class):
    df = pd.read_csv(filepath)

    anchors_df = df.groupby(['label']).head(n_samples_per_class).reset_index(drop=True)

    triplets = []
    for _, anchor in anchors_df.iterrows():
        pos_samples = df.loc[(df["label"] == anchor["label"]) & (df["id"] != anchor["id"])].sample(3)
        neg_samples = df.loc[df["label"] != anchor["label"]].groupby('label').sample(1)

        for i in range(n_labels - 1):
            triplets.append({
                "anchor": anchor["audio_path"],
                "positive": pos_samples["audio_path"].iloc[i],
                "negative": neg_samples["audio_path"].iloc[i],
                "anchor_label": anchor["label"],
                "positive_label": pos_samples["label"].iloc[i],
                "negative_label": neg_samples["label"].iloc[i],
            })

    triplet_df = pd.DataFrame(triplets)
    triplet_df.to_csv(f"../data/triplets_{n_labels}c_{len(triplets)}.csv", index=False)


if __name__ == '__main__':
    # create_triplet_dataset('../data/crowd_train_2000.csv', N_LABELS, N_SAMPLES_PER_CLASS)
    create_triplet_dataset('../data/crowd_train_bin_4000.csv', 2, 300)
