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
    isXLColName: str = "NuXL:isXL",         # str: Column name indicating if the peptide is crosslinked
    label_col: str = "Label"                # str: Column name for the label, indicating target (1) or decoy
):
    scores.sort_values("Score",ascending=False,inplace=True)
    if not set([isXLColName]).issubset(scores.columns):
        compute_ID_no(scores, label_col, 'q-val')
    else:
        scores = scores.groupby(isXLColName)\
        .apply(
            lambda x: compute_ID_no(x, label_col, "class-specific_q-val")
        )
        scores.index = scores.index.droplevel(isXLColName)
    return scores

def read_top_down_data(input_file):
    data = pd.read_csv(input_file, sep="\t")
    data['Label'] = (data['ProteinAccession'].str.contains("DECOY")==False).astype(float)
    return data

def rerank_helper(x, score_col, new_rank_col):
    x.sort_values(score_col,ascending=False,inplace=True)
    x[new_rank_col] = range(1,len(x) + 1)
    return x

def rerank(df, group_col, score_col, new_rank_col):
    # rerank PSMs
    df = df.groupby(group_col)\
        .apply(
            lambda x: rerank_helper(x, score_col, new_rank_col)
        )
    df.index = df.index.droplevel(group_col)
    return df

def get_datasets():
    datasets = {1: {'type': 'crosslink_data',
                    'file':'AChernev_080219_HeLa_RNA_UV',
                    'file_ending': '.idXML',
                    'name':'AChernev_080219',
                    'group': 'NuXL:isXL',
                    'comparison':'opti_'},
                2: {'type': 'crosslink_data',
                    'file':'M_Raabe_A_Wulf_220421_270421_Expl3_Ecoli_XL_UV_S30_LB_bRPfrac_11',
                    'file_ending': '.idXML',
                    'name':'M_Raabe_A_Wulf_220421_270421',
                    'group': 'NuXL:isXL',
                    'comparison':'perc'},
                3: {'type': 'crosslink_data',
                    'file':'M_Raabe_A_Wulf_220421_290421_Expl3_Ecoli_XL_DEB_S30_LB_bRPfrac_12',
                    'file_ending': '.idXML',
                    'name':'M_Raabe_A_Wulf_220421_290421',
                    'group': 'NuXL:isXL',
                    'comparison':'perc'},
                4: {'type': 'crosslink_data',
                    'file':'MRaabe_LW_091221_171221_Expl2_XL_Ecoli_NM_S30_bRP_rep1_11',
                    'file_ending': '.idXML',
                    'name':'MRaabe_LW_091221_171221',
                    'group': 'NuXL:isXL',
                    'comparison':'perc'},
                5: {'type': 'top_down_data',
                    'file':'outprsm4_multihits',
                    'file_ending': '.tsv',
                    'name':'outpsrm4',
                    'group': 'ModCount',
                    'comparison': None}}
    return datasets