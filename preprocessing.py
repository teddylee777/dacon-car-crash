import pandas as pd
import os


def load(root, data_dir, filename):
    df = pd.read_csv(os.path.join(root, data_dir, filename))
    df['video_path'] = df['video_path'].apply(lambda x: os.path.join(root, data_dir, x[2:]))
    return df

def convert_weather(x):
    if x in [1, 2, 7, 8]:
        return 0
    elif x in [3, 4, 9, 10]:
        return 1
    elif x in [5, 6, 11, 12]:
        return 2

def create_features(df):
    # (crash) 0: no, 1: yes
    df.loc[df['label'] == 0, 'crash'] = 0
    df.loc[df['label'] != 0, 'crash'] = 1
    
    # (ego) 0: yes, 1: no
    df['ego'] = df['label'].apply(lambda x: 0 if x < 7 else 1)
    
    # (weather) 0: normal, 1: snowy, 2: rainy
    df['weather_normal'] = df['label'].apply(lambda x: x in [1, 2, 7, 8])
    df['weather_snowy'] = df['label'].apply(lambda x: x in [3, 4, 9, 10])
    df['weather_rainy'] = df['label'].apply(lambda x: x in [5, 6, 11, 12])
    df['weather'] = df['label'].apply(convert_weather)
    
    # (timing) 0: day, 1: night
    df['timing'] = df['label'].apply(lambda x: int(x % 2 == 0))
    return df






