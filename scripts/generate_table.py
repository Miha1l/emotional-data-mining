import pandas as pd
import json
import math


def emo_2_label(emotion):
    d = {
        "neutral": 0,
        "angry": 1,
        "positive": 2,
        "sad": 3
    }

    return d[emotion]


def generate_table(label):
    filepath = f"../dusha/crowd/crowd_{label}/raw_crowd_{label}.jsonl"
    l = []
    with open(filepath) as f:
        for line in f:
            data = json.loads(line)
            id = data["hash_id"]

            if isinstance(data["speaker_emo"], float) and math.isnan(data["speaker_emo"]):
                continue

            row = {
                "id": id,
                "audio_path": f"dusha/crowd/crowd_{label}/" + data["audio_path"],
                "features_path": "dusha/features/" + f"{id}.npy",
                "emotion": data["speaker_emo"],
                "label": emo_2_label(data["speaker_emo"]),
                "duration": data["duration"]
            }

            l.append(row)

    df = pd.DataFrame(l)
    df = df.drop_duplicates(subset=["id"])
    df.to_csv(f"../data/crowd_{label}.csv", index=False)


TRAIN = "train"
TEST = "test"

if __name__ == "__main__":
    generate_table(TRAIN)
