import numpy as np
import pandas as pd
from q_value_calc_crosslinks import readAndProcessIdXML, calcQ, FDR_summary
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

#input_file = "data/crosslink_data/M_Raabe_A_Wulf_220421_270421_Expl3_Ecoli_XL_UV_S30_LB_bRPfrac_11.idXML" # rank 0
input_file = "data/crosslink_data/M_Raabe_A_Wulf_220421_270421_Expl3_Ecoli_XL_UV_S30_LB_bRPfrac_11_perc.idXML" # rank 0 - 7
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
df = readAndProcessIdXML(input_file)
# filter data and sort according to score
df = df.filter(['Label', 'Score','peplen', 'NuXL:isXL', 'NuXL:modds', 'NuXL:pl_modds', 
                'NuXL:mass_error_p', 'NuXL:tag_XLed', 'NuXL:tag_unshifted' ,
                'NuXL:tag_shifted', 'missed_cleavages', 'NuXL:ladder_score',
                'variable_modifications', 'rank'])
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
    grid_search.fit(df.loc[train_idx, df.columns != 'Label'], np.ravel(df.loc[train_idx, df.columns == 'Label']))
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.3f}")
    
    # run model on all data points
    probabilities = grid_search.predict_proba(df.loc[:, df.columns != 'Label'])
    print(f"Predicting class probabilities: {probabilities}") # might not be correct because of Platt's method
    
    # features of best model
    print(f"Feature weights: {grid_search.best_estimator_.coef_}")
    feature_scaling = pd.DataFrame(np.vstack([scaler.data_min_,scaler.data_max_]).T, columns = ["min", "max"], index=features)
    print(f"Feature scaling:{feature_scaling}")
    
    # set probabilites as new score
    new_scores = pd.DataFrame(np.vstack([probabilities[:,1], df['Label'], df['NuXL:isXL']]).T, columns = ['Score', 'Label', 'NuXL:isXL'])
    
    # q value computation for new scores
    q_vals = calcQ(new_scores)
    FDR_summary(q_vals)
    q_vals.to_csv(input_file + ".qvals")