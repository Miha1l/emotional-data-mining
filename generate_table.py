import json
import numpy as np
import pandas as pd
import click
from os import path


@click.command()
@click.option("--filepath", "filepath", required=True, help="Path to jsonl file", type=click.Path(exists=True))
@click.option("--mean", "mean", is_flag=True, required=False, default=False, help="Calculate the average values of the attributes for each record")
@click.option("--output", "output", required=False, default="output.csv", help="Name of output file")
def generate_csv_features_table(filepath, mean, output):
    if filepath.rsplit('.', 1)[-1].lower() != 'jsonl':
        print("Incorrect file type. Required .jsonl")
        return
    df = pd.DataFrame()
    with open(filepath) as f:
        for line in f:
            data = json.loads(line)
            features = np.load(path.normpath(filepath + '/../' + data['tensor']))[0]
            features_df = pd.DataFrame(features).T
            if mean:
                features_df = features_df.mean(axis=0).to_frame().T
            features_df['label'] = data['label']
            features_df['emotion'] = data['emotion']
            df = pd.concat([df, features_df], axis=0)
    df.to_csv(output, index=False)


if __name__ == '__main__':
    generate_csv_features_table()