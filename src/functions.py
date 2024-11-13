import pandas as pd
import numpy as np

def compute_ID_no(df: pd.DataFrame, label_col, q_val_col):
    df.sort_values([q_val_col, 'Score', label_col], ascending=[True, False, False], inplace=True)
    df['cum_target_id'] = np.cumsum(df[label_col])
    df = df.groupby(q_val_col)\
    .apply(
        lambda x: compute_max_ID_no_per_q_val(x, label_col, q_val_col)
    )
    df.index = df.index.droplevel(q_val_col)
    return df

def compute_max_ID_no_per_q_val(df, label_col, q_val_col):
    df.sort_values([q_val_col, 'Score', label_col], ascending=[True, False, False], inplace=True)
    df['cum_target_id'] = np.ones(len(df['cum_target_id']))*np.max(df['cum_target_id'])
    return df

def get_target_id(
    scores: pd.DataFrame,                   # pd.DataFrame: Input data containing PSMs and q-values
    isXLColName: str = "NuXL:isXL",           # str: Column name indicating if the peptide is crosslinked
    label_col: str = "Label",              # str: Column name for the label, indicating target (1) or decoy
    q_val_col: str = "class-specific_q-val"
):
    scores.sort_values("Score",ascending=False,inplace=True)
    scores = scores.groupby(isXLColName)\
    .apply(
        lambda x: compute_ID_no(x, label_col, q_val_col)
    )
    scores.index = scores.index.droplevel(isXLColName)
    return scores

def parse_top_down_data(input_file):
    data = pd.read_csv(input_file, sep="\t")
    return data