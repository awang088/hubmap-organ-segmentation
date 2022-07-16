import pandas as pd
from sklearn.model_selection import GroupKFold

def get_fold_idx_list(train_df, fold, seed):
    folds = list(GroupKFold(n_splits=5, random_state=seed, shuffle=True).split(train_df, groups=train_df['patient_id']))
    trn_idxs_list = []
    val_idxs_list = []
    for _fold, (train_idxs, val_idxs) in enumerate(folds):
        if _fold == fold:
            val_idxs_list = val_idxs
            trn_idxs_list = train_idxs
    return trn_idxs_list, val_idxs_list
