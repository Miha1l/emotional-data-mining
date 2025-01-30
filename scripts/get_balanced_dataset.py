import pandas as pd


def create_balanced_df(dataset_path, n_samples, save_path):
    df = pd.read_csv(dataset_path)
    df_part = df.groupby(['emotion']).head(n_samples).reset_index(drop=True)
    df_part.to_csv(save_path, index=False)


if __name__ == '__main__':
    n_samples = 125
    create_balanced_df('../data/crowd_train.csv', n_samples, f'../data/crowd_train_{n_samples}.csv')
