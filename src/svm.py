import numpy as np
import pandas as pd
from q_value_calc_crosslinks import readAndProcessIdXML, calcQ, FDR_summary
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

input_file = "data/crosslink_data/M_Raabe_A_Wulf_220421_270421_Expl3_Ecoli_XL_UV_S30_LB_bRPfrac_11.idXML" # rank 0
#input_file = "data/crosslink_data/M_Raabe_A_Wulf_220421_270421_Expl3_Ecoli_XL_UV_S30_LB_bRPfrac_11_perc.idXML" # rank 0 - 7
#input_file = "data/crosslink_data/M_Raabe_A_Wulf_220421_290421_Expl3_Ecoli_XL_DEB_S30_LB_bRPfrac_12.idXML" # rank 0
#input_file = "data/crosslink_data/M_Raabe_A_Wulf_220421_290421_Expl3_Ecoli_XL_DEB_S30_LB_bRPfrac_12_perc.idXML" # rank 0 - 7
#input_file = "data/crosslink_data/MRaabe_LW_091221_171221_Expl2_XL_Ecoli_NM_S30_bRP_rep1_11.idXML" # rank 0
#input_file = "data/crosslink_data/MRaabe_LW_091221_171221_Expl2_XL_Ecoli_NM_S30_bRP_rep1_11_perc.idXML" # rank 0 - 7

features = ['Score','peplen', 'NuXL:isXL', 'NuXL:modds', 'NuXL:pl_modds', 
                'NuXL:mass_error_p', 'NuXL:tag_XLed', 'NuXL:tag_unshifted' ,
                'NuXL:tag_shifted', 'missed_cleavages', 'NuXL:ladder_score',
                'variable_modifications', 'rank']

# this is not used because this is exactly what .coef_ does
def getFeatureWeights(model:SVC):
    feature_weights = dict(zip(features,np.zeros(len(features))))
    for i,sv in enumerate(model.support_vectors_):
        sv_coef = model.dual_coef_[0][i]
        for j,feature in enumerate(sv):
            feature_weights[features[j]] += sv_coef * feature
    return feature_weights

# read data
original_df = readAndProcessIdXML(input_file)

q_vals_before = calcQ(original_df)
# filter data and sort according to score
df = original_df.filter(features)
df = df.sort_values('Score',ascending=False)

# determine minority class 
minority_class = min({df.loc[df['NuXL:isXL'] == 0].size, df.loc[df['NuXL:isXL'] == 1].size})
if (minority_class > 500): minority_class = 500
# define training data (peptides with top and bottom scores of each class with rank 0)
pep_top = df.loc[(df['NuXL:isXL'] == 0) & (df['rank'] == 0)][:int(minority_class/2)]
pep_bottom = df.loc[(df['NuXL:isXL'] == 0) & (df['rank'] == 0)][-int(minority_class/2):]
XL_top = df.loc[(df['NuXL:isXL'] == 1) & (df['rank'] == 0)][:int(minority_class/2)]
XL_bottom = df.loc[(df['NuXL:isXL'] == 1) & (df['rank'] == 0)][-int(minority_class/2):]
train_idx = np.concatenate([pep_top.index, pep_bottom.index, XL_top.index, XL_bottom.index])

labels = np.empty(len(df))
labels[pep_bottom.index] = 0
labels[XL_bottom.index] = 0
labels[pep_top.index] = 1
labels[XL_top.index] = 1

# min-max feature scaling to 0-1
scaler = MinMaxScaler()
scaler.fit(df.loc[train_idx,features])
df[features] = scaler.transform(df[features])

if (minority_class > 10):
    # grid search on C value
    param_grid = {
        'C': np.power(float(2), [-5,-1,1,5,7,11,15]),
    }
    grid_search = GridSearchCV(
        SVC(kernel='linear', probability=True),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    # fit model to train data
    grid_search.fit(df.loc[train_idx, :], labels[train_idx])
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.3f}")
    
    # run model on all data points
    probabilities = grid_search.predict_proba(df.loc[:, :])
    print(f"Predicting class probabilities: {probabilities}") # might not be correct because of Platt's method
    
    # features of best model
    print(f"Feature weights: {grid_search.best_estimator_.coef_}")
    feature_scaling = pd.DataFrame(np.vstack([scaler.data_min_,scaler.data_max_]).T, columns = ["min", "max"], index=features)
    print(f"Feature scaling:{feature_scaling}")
    
    # set probabilites as new score
    new_scores = pd.DataFrame(np.vstack([probabilities[:,1], original_df['Label'], original_df['NuXL:isXL']]).T, columns = ['Score', 'Label', 'NuXL:isXL'])
    
    # q value computation for new scores
    q_vals_after = calcQ(new_scores)


from pyopenms import *

def get_target_id(
    scores: pd.DataFrame,                   # pd.DataFrame: Input data containing PSMs and q-values
    isXLColName: str = "NuXL:isXL",           # str: Column name indicating if the peptide is crosslinked
    label_col: str = "Label",              # str: Column name for the label, indicating target (1) or decoy
    q_val_col: str = "class-specific_q-val",
    max_qvalue: float = 0.1
):
    scores.sort_values(q_val_col,ascending=True,inplace=True)
    # Initialize a dictionary to store counts of PSMs below each q-value threshold
    result = []
    for XL in [0,1]:
        XL_summary_q_val_count = []
        
        # Filter for crosslinked (XL) PSMs
        XL_psms_q_df = scores[scores[isXLColName] == XL]

        XL_q_value_thresholds = XL_psms_q_df[q_val_col]
        # Further filter to keep only target PSMs (label == 1)
        XL_psms_q_df = XL_psms_q_df[XL_psms_q_df[label_col] == 1]
        
        # Count the number of PSMs below each q-value threshold
        for qvalue in XL_q_value_thresholds:
            # Count XL PSMs below the class-specific q-value threshold
            XL_summary_q_val_count.append(sum(j < min(qvalue, max_qvalue) for j in list(XL_psms_q_df[q_val_col])))
        result.append(XL_q_value_thresholds)
        result.append(XL_summary_q_val_count)

    return result

pep_q_value_thresholds_after, pep_target_id_after, XL_q_value_thresholds_after, XL_target_id_after = get_target_id(q_vals_after)
pep_q_value_thresholds_before, pep_target_id_before, XL_q_value_thresholds_before, XL_target_id_before  = get_target_id(q_vals_before)

fig, axs = plt.subplots(2)
axs[0].step(XL_q_value_thresholds_after,XL_target_id_after, label="SVM" )
axs[1].step(pep_q_value_thresholds_after,pep_target_id_after, label="SVM")
axs[0].step(XL_q_value_thresholds_before,XL_target_id_before, label="no SVM" )
axs[1].step(pep_q_value_thresholds_before,pep_target_id_before, label = "no SVM")
axs[0].set_xlim(0,0.1)
axs[1].set_xlim(0,0.1)
axs[0].set_title("XL")
axs[1].set_title("Peptide")
axs[0].set_xlabel("q-value")
axs[1].set_xlabel("q-value")
axs[1].set_ylabel("no. target IDs")
axs[0].set_ylabel("no. target IDs")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
fig.tight_layout()
plt.show()

