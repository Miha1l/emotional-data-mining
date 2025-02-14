import pandas as pd
import datetime


def create_balanced_df(dataset_path, n_samples, save_path):
    df = pd.read_csv(dataset_path)
    df_part = df.groupby(['emotion']).head(n_samples).reset_index(drop=True)
    df_part.to_csv(save_path, index=False)
    print(f"Суммарное время аудио: {datetime.timedelta(seconds=df_part['duration'].sum())}")


if __name__ == '__main__':
    n_samples = 750
    dataset_name = 'crowd_train'
    create_balanced_df(f'../data/{dataset_name}.csv', n_samples, f'../data/{dataset_name}_{n_samples}.csv')
