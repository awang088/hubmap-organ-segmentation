import sys
import os

sys.path.append(os.path.abspath("/home/andrew/Documents/HuBMAP/src"))
import warnings
warnings.simplefilter('ignore')
from utils import fix_seed
from get_config import get_config
from get_fold_idx_list import get_fold_idx_list
from run import run
import pickle

import numpy as np
import pandas as pd
from os.path import join as opj
from sklearn.model_selection import GroupKFold


if __name__=='__main__':
    # config
    fix_seed(42)
    config = get_config()
    FOLD_LIST = config['FOLD_LIST']
    VERSION = config['VERSION']
    INPUT_PATH = config['INPUT_PATH']
    OUTPUT_PATH = config['OUTPUT_PATH']
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    device = config['device']
    print(device)
    
    # dataset
    data_df = []
    for data_path in config['train_data_path_list']:
        _data_df = pd.read_csv(opj(data_path,'data.csv'))
        _data_df['data_path'] = data_path
        data_df.append(_data_df)
    data_df = pd.concat(data_df, axis=0).reset_index(drop=True)

    print('data_df.shape = ', data_df.shape)
    data_df = data_df[data_df['std_img']>10].reset_index(drop=True)
    print('data_df.shape = ', data_df.shape)
    data_df['binned'] = np.round(data_df['ratio_masked_area'] * config['multiplier_bin']).astype(int)
    data_df['is_masked'] = data_df['binned']>0

    trn_df = data_df.copy()
    trn_df['binned'] = trn_df['binned'].apply(lambda x:config['binned_max'] if x>=config['binned_max'] else x)
    trn_df_1 = trn_df[trn_df['is_masked']==True]
    print(trn_df['is_masked'].value_counts())
    print(trn_df_1['binned'].value_counts())
    print('mean = ', int(trn_df_1['binned'].value_counts().mean()))
    
    data_df['patient_id'] = data_df['filename_img'].apply(lambda x:x.split('_')[0])

    # train
    for seed in config['split_seed_list']:
        kfold = list(GroupKFold(n_splits=5).split(data_df, groups=data_df['patient_id']))
        run(seed, data_df, None, kfold)
        
    # score
    score_list  = []
    for seed in config['split_seed_list']:
        for fold in config['FOLD_LIST']:
            log_df = pd.read_csv(opj(config['OUTPUT_PATH'],f'log_seed{seed}_fold{fold}.csv'))
            score_list.append(log_df['val_score'].max())
    print('CV={:.4f}'.format(sum(score_list)/len(score_list)))