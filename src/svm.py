import numpy as np
from q_value_calc_crosslinks import readAndProcessIdXML
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

#input_file = "data/crosslink_data/M_Raabe_A_Wulf_220421_270421_Expl3_Ecoli_XL_UV_S30_LB_bRPfrac_11.idXML" # rank 0
#input_file = "data/crosslink_data/M_Raabe_A_Wulf_220421_270421_Expl3_Ecoli_XL_UV_S30_LB_bRPfrac_11_perc.idXML" # rank 0 - 7
#input_file = "data/crosslink_data/M_Raabe_A_Wulf_220421_290421_Expl3_Ecoli_XL_DEB_S30_LB_bRPfrac_12.idXML" # rank 0
#input_file = "data/crosslink_data/M_Raabe_A_Wulf_220421_290421_Expl3_Ecoli_XL_DEB_S30_LB_bRPfrac_12_perc.idXML" # rank 0 - 7
#input_file = "data/crosslink_data/MRaabe_LW_091221_171221_Expl2_XL_Ecoli_NM_S30_bRP_rep1_11.idXML" # rank 0
input_file = "data/crosslink_data/MRaabe_LW_091221_171221_Expl2_XL_Ecoli_NM_S30_bRP_rep1_11_perc.idXML" # rank 0 - 7


df = readAndProcessIdXML(input_file)
df = df.filter(['Label', 'Score','peplen', 'NuXL:isXL', 'NuXL:modds', 'NuXL:pl_modds', 
                'NuXL:mass_error_p', 'NuXL:tag_XLed', 'NuXL:tag_unshifted' ,
                'NuXL:tag_shifted', 'missed_cleavages', 'NuXL:ladder_score',
                'variable_modifications', 'rank'])
df = df.sort_values('Score',ascending=False)
minority_class = min({df.loc[df['NuXL:isXL'] == 0].size, df.loc[df['NuXL:isXL'] == 1].size})
if (minority_class > 500): minority_class = 500
# top and bottom of each class with rank 0
pep_top = df.loc[(df['NuXL:isXL'] == 0) & (df['rank'] == 0)][:int(minority_class/2)]
pep_bottom = df.loc[(df['NuXL:isXL'] == 0) & (df['rank'] == 0)][-int(minority_class/2):]
XL_top = df.loc[(df['NuXL:isXL'] == 1) & (df['rank'] == 0)][:int(minority_class/2)]
XL_bottom = df.loc[(df['NuXL:isXL'] == 1) & (df['rank'] == 0)][-int(minority_class/2):]
train_idx = np.concatenate([pep_top.index, pep_bottom.index, XL_top.index, XL_bottom.index])

if (minority_class > 10):
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
    grid_search.fit(df.loc[train_idx, df.columns != 'Label'], np.ravel(df.loc[train_idx, df.columns == 'Label']))
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.3f}")
    probabilities = grid_search.predict_proba(df.loc[:, df.columns != 'Label'])
    print(probabilities)

    #https://openms.de/current_doxygen/html/classOpenMS_1_1SimpleSVM.html#a89fed348bb9d07bc5ada42f74b4500c1
