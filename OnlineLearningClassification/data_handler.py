import json
import sys

import pandas as pd
import tqdm

from utils import print_df_to_table

SUBJECT_PATH = 'Data/Subject_details.csv'
VIDEO_PATH = 'Data/Video_details.csv'
EEG_PATH = 'Data/EEG_data.csv'


def show_count(df, class_col):
    print('[INFO] {0} Distribution'.format(class_col.title()))
    cvc = df[class_col].value_counts(sort=False)
    sdf = cvc.to_frame()
    sdf.insert(0, 'Class', cvc.index)
    sdf.columns = [class_col.title(), 'Count']
    print_df_to_table(sdf)
    return df


def replace_categorical_cols(df, except_col):
    cat_cols = list(set(df.columns) - set(df._get_numeric_data().columns))
    for col in cat_cols:
        if col == except_col:
            continue
        print('[INFO] Replacing Categorical Values in Column :: {0}'.format(col))
        show_count(df, col)
        rpd = {v: k + 1 for k, v in enumerate(sorted(df[col].unique()))}
        df[col].replace(rpd, inplace=True)
        with open('Data/{0}.json'.format(col.replace('/', '(or)')), 'w') as f:
            json.dump(rpd, f, indent=4, sort_keys=False)
    df.reset_index(drop=True, inplace=True)
    return df


def load_data():
    print('[INFO] Loading Subject Details :: {0}'.format(SUBJECT_PATH))
    s_df = pd.read_csv(SUBJECT_PATH)
    print('[INFO] Data Shape :: {0}'.format(s_df.shape))
    print('[INFO] Loading VIDEO Details :: {0}'.format(VIDEO_PATH))
    v_df = pd.read_csv(VIDEO_PATH)
    print('[INFO] Data Shape :: {0}'.format(v_df.shape))
    print('[INFO] Loading EEG DATA :: {0}'.format(EEG_PATH))
    e_df = pd.read_csv(EEG_PATH)
    print('[INFO] Data Shape :: {0}'.format(e_df.shape))
    return s_df, v_df, e_df


def merge_data(s_df, v_df, e_df):
    vid_data = []
    sub_data = []
    print('[INFO] Mergin')
    for row in tqdm.tqdm(e_df.values, desc='[INFO] Merging Video and Subject Data :', file=sys.stdout):
        sub_v_df = v_df[v_df['Video_ID'] == int(row[0])].values[0]
        vid_data.append([
            sub_v_df[0], sub_v_df[1], sub_v_df[-1]
        ])
        sub_s_df = s_df[s_df['Subject_ID'] == int(row[1])].values[0]
        sub_data.append([
            sub_s_df[0], sub_s_df[1], sub_s_df[2], sub_s_df[4]
        ])

    print('[INFO] Merging Data')
    eeg_cols = list(e_df.columns)[2:-1]
    vid_cols = ['Video_ID', 'Video_Title', 'Video_Instructor']
    sub_cols = ['Subject_ID', 'Subject_Gender', 'Subject_Age', 'Subject_FOI']
    e_df[vid_cols] = vid_data
    e_df[sub_cols] = sub_data
    e_df['Understand'] = e_df['subject_understood']
    df = e_df[sub_cols + vid_cols + eeg_cols + ['Understand']]
    print('[INFO] Data Shape After Merged :: {0}'.format(df.shape))
    dp = 'Data/merged.csv'
    print('[INFO] Saving Merged Data :: {0}'.format(dp))
    df.to_csv(dp, index=False)
    return df


def preprocess_data(df):
    print('INFO] PreProcessing')
    df = replace_categorical_cols(df, 'Understand')
    show_count(df, 'Understand')
    dp = 'Data/preprocessed.csv'
    print('[INFO] Saving Preprocessed Data :: {0}'.format(dp))
    df.to_csv(dp, index=False)
    return df


if __name__ == '__main__':
    preprocess_data(merge_data(*load_data()))
