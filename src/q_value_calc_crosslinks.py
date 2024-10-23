import pandas as pd
from pyopenms import *
from typing import Tuple, Dict

# convert every string col into an int or float if possible
def strToFloat(df):
  for col in df:
    try:
      df[col] = [float(i) for i in df[col]]
    except ValueError:
      continue
  return df# convert every string col into an int or float if possible


def readAndProcessIdXML(input_file, top=1):
  """
  convert the (.idXML) format identification file to dataframe
  """
  prot_ids = []; pep_ids = []
  IdXMLFile().load(input_file, prot_ids, pep_ids)
  meta_value_keys = []
  rows = []
  for peptide_id in pep_ids:
    spectrum_id = peptide_id.getMetaValue("spectrum_reference")
    scan_nr = spectrum_id[spectrum_id.rfind('=') + 1 : ]

    hits = peptide_id.getHits()

    psm_index = 1
    for h in hits:
      if psm_index > top:
        break
      charge = h.getCharge()
      score = h.getScore()
      z2 = 0; z3 = 0; z4 = 0; z5 = 0

      if charge == 2:
          z2 = 1
      if charge == 3:
          z3 = 1
      if charge == 4:
          z4 = 1
      if charge == 5:
          z5 = 1
      if "target" in h.getMetaValue("target_decoy"):
          label = 1
      else:
          label = 0
      sequence = h.getSequence().toString()
      if len(meta_value_keys) == 0: # fill meta value keys on first run
        h.getKeys(meta_value_keys)
        meta_value_keys = [x.decode() for x in meta_value_keys]
        all_columns = ['SpecId','PSMId','Label','Score','ScanNr','Peptide','peplen','ExpMass','charge2','charge3','charge4','charge5','accessions'] + meta_value_keys
        #print(all_columns)
      # static part
      accessions = ';'.join([s.decode() for s in h.extractProteinAccessionsSet()])

      row = [spectrum_id, psm_index, label, score, scan_nr, sequence, str(len(sequence)), peptide_id.getMZ(), z2, z3, z4, z5, accessions]
      # scores in meta values
      for k in meta_value_keys:
        s = h.getMetaValue(k)
        if type(s) == bytes:
          s = s.decode()
        row.append(s)
      rows.append(row)
      psm_index += 1
      break; # parse only first hit
  
  df =pd.DataFrame(rows, columns=all_columns)
  convert_dict = {'SpecId': str,
                  'PSMId': int,
                  'Label': int,
                  'Score': float,
                  'ScanNr': int,
                  'peplen': int                
               }
  
  df = df.astype(convert_dict)
  return df


def calcQ(
    df: pd.DataFrame,                    # DataFrame: The input data containing scores and labels
    scoreColName: str = 'Score',         # str: The column name containing the score values
    labelColName: str = 'Label',         # str: The column name indicating whether the row is a target (1) or decoy (0 or -1)
    isXLColName: str = 'NuXL:isXL',      # str: The column name indicating crosslinked peptides (1 for crosslinked, 0 for not)
    addXlQ: bool = True,                 # bool: Whether to calculate class-specific q-values for crosslinked and non-crosslinked peptides
    ascending: bool = False,             # bool: Sorting order for the score column (False for descending)
    remove_decoy: bool = False           # bool: Whether to remove decoy entries from the resulting DataFrame
) -> pd.DataFrame:                       # Returns a DataFrame with q-values and optionally class-specific q-values
    
    # Check if required columns are present in the DataFrame
    if not set([scoreColName, labelColName, isXLColName]).issubset(df.columns):
        raise Exception("column missing")

    # Sort the DataFrame by the score column
    df.sort_values(scoreColName, ascending=ascending, inplace=True)

    # Replace -1 in the label column with 0
    df[labelColName].replace(to_replace=-1, value=0, inplace=True)
    
    # Calculate the FDR (False Discovery Rate)
    df['FDR'] = (range(1, len(df) + 1) / df[labelColName].cumsum()) - 1

    # Calculate the q-value (minimum FDR for each score and above)
    df['q-val'] = df['FDR'][::-1].cummin()[::-1]
    
    # Optionally, calculate class-specific q-values for crosslinked (XL) and non-crosslinked peptides
    if addXlQ:
        ls = []
        for XL in [0, 1]:
            # Split the DataFrame based on whether the peptide is crosslinked
            currClass = pd.DataFrame(df[df[isXLColName] == XL])
            ls.append(currClass)
            
            # Calculate class-specific FDR and q-value
            currClass.sort_values(scoreColName, ascending=ascending, inplace=True)
            FDR = (range(1, len(currClass[labelColName]) + 1) / currClass[labelColName].cumsum()) - 1
            currClass['class-specific_q-val'] = FDR[::-1].cummin()[::-1]
            
        # Combine the DataFrames for crosslinked and non-crosslinked peptides
        df = pd.concat(ls)
        df.sort_values(scoreColName, ascending=ascending, inplace=True)

    # Optionally, remove decoy rows (label == 0)
    if remove_decoy:
        df = df[df[labelColName] == 1]
       
    return df


def FDR_summary(
    psm_q_df: pd.DataFrame,                   # pd.DataFrame: Input data containing PSMs and q-values
    isXLColName: str = "NuXL:isXL",           # str: Column name indicating if the peptide is crosslinked
    label_col: str = "Label",                 # str: Column name for the label, indicating target (1) or decoy
    q_val_col: str = "class-specific_q-val",  # Union[str, List[str]]: Column(s) containing q-values
    q_value_thresholds: List[float] = [0.01, 0.05, 0.1]           # List[float]: List of q-value thresholds to summarize
) -> Tuple[pd.DataFrame, Dict[float, int]]:   # Returns a tuple with filtered DataFrame and a dictionary of q-value counts
    
    # Initialize a dictionary to store counts of PSMs below each q-value threshold
    summary_q_val_count = {}
    
    # Filter for crosslinked (XL) PSMs
    XL_psms_q_df = psm_q_df[psm_q_df[isXLColName] == 1]

    # Further filter to keep only target PSMs (label == 1)
    XL_psms_q_df = XL_psms_q_df[XL_psms_q_df[label_col] == 1]

    # Count the number of PSMs below each q-value threshold
    for qvalue in q_value_thresholds:
        # Count XL PSMs below the class-specific q-value threshold
        summary_q_val_count[qvalue] = sum(j < qvalue for j in list(XL_psms_q_df[q_val_col]))

    # Print the counts for each q-value column
    print(q_val_col, " counts: ", summary_q_val_count)

    # Return the filtered DataFrame and the summary counts
    return XL_psms_q_df, summary_q_val_count


### calculate crosslink peptide sequence level
def calc_xl_peptide_qval(
    df: pd.DataFrame,                      # pd.DataFrame: The input DataFrame containing the data
    scoreColName: str = 'Score',           # str: Column name for score values
    seq_col: str = "Peptide",              # str: Column name for the peptide sequences
    labelColName: str = 'Label',           # str: Column name for the label, indicating target (1) or decoy (0 or -1)
    isXLColName: str = 'NuXL:isXL',        # str: Column name indicating if the peptide is crosslinked
    ascending: bool = False,               # bool: Whether to sort the scores in ascending order (False for descending)
    remove_decoy: bool = True              # bool: Whether to remove decoy entries (where label != 1)
) -> pd.DataFrame:                         # Returns a DataFrame with q-values for crosslinked peptides

    # Keep only the crosslinked (XL) peptides
    XL_df = df[df[isXLColName] == 1]

    # Remove modifications from the peptide sequence (convert to unmodified sequence)
    XL_df.loc[:, "sequence"] = XL_df[seq_col].replace(r"\([^)]*\)", '', regex=True)

    # Find the maximum score for each sequence
    df_max_score = XL_df.groupby('sequence')[scoreColName].transform('max')

    # Keep only the rows where the score is equal to the maximum score for each sequence
    XL_df = XL_df[XL_df[scoreColName] == df_max_score]

    # Calculate the q-value for the crosslinked peptides
    pept_level_xl_q = calcQ(
        XL_df, 
        scoreColName=scoreColName, 
        labelColName=labelColName, 
        isXLColName=isXLColName, 
        addXlQ=False, # all are crosslink peptide just one class
        ascending=ascending, 
        remove_decoy=remove_decoy
    )

    return pept_level_xl_q

## calculate xl protein-level q-val at unique peptides 
def calculate_xl_prt_qval(
    df: pd.DataFrame,                  # DataFrame: Input data
    score_col: str = "Score",          # str: Column name for score values
    seq_col: str = "Peptide",          # str: Column name for peptide sequences
    prot_col: str = 'accessions',      # str: Column name for protein accessions
    label_col: str = "Label",          # str: Column name for label information
    isXLColName: str = "NuXL:isXL",    # str: Column name indicating crosslinked peptides
    remove_decoy: bool = True,         # bool: Whether to remove decoy entries
    ascending: bool = False            # bool: Sorting order, ascending if True, descending if False
) -> Tuple[pd.DataFrame, pd.DataFrame]:  # Tuple of DataFrames: full data and filtered q-value==0.01 data
    
    # Keep only the crosslinked (XL) peptides
    XL_df = df[df[isXLColName] == 1]

    # Determine whether the protein is unique (no semicolon in accessions)
    XL_df['is_unique'] = ~(XL_df[prot_col].str.contains(';'))

    # Filter for unique proteins
    XL_df_un = XL_df[XL_df['is_unique'] == True]

    # Remove modifications from the peptide sequence (convert to unmodified sequence)
    XL_df.loc[: , "sequence"] = XL_df[seq_col].replace(r"\([^)]*\)", '', regex=True)

    # Find the maximum score for each sequence
    df_max_score = XL_df.groupby('sequence')[score_col].transform('max')

    # Keep rows where the score is the maximum for each sequence
    XL_df = XL_df[XL_df[score_col] == df_max_score]

    # Sort by score and drop duplicates, keeping the highest score for each protein
    XL_df_un_highest_score = XL_df_un.sort_values(by=score_col, ascending=False).drop_duplicates(subset=prot_col, keep='first')

    # Calculate the q-value
    XL_df_un_highest_score_q = calcQ(
        XL_df_un_highest_score, 
        scoreColName=score_col, 
        labelColName=label_col, 
        isXLColName=isXLColName, 
        addXlQ=False, # all are crosslink peptide just one class
        ascending=ascending, 
        remove_decoy=remove_decoy
    )

    # Return the DataFrame with q-values and a filtered DataFrame where q-value is below 0.01
    return XL_df_un_highest_score_q, XL_df_un_highest_score_q[XL_df_un_highest_score_q["q-val"] < 0.01]

"""
## Analyze one file
file_name = "MRaabe_LW_091221_171221_Expl2_XL_Ecoli_NM_S30_bRP_rep1_11_perc.idXML"
nuxl_out_df = readAndProcessIdXML(file_name, top=1) ## consider 1 top hit
nuxl_out_df.to_csv(file_name.split('.')[0]+"_idxml_to_csv.csv") # check top hit while saving

## CSM-level FDR
print("\n --- CSM and PSM Level FDR ----")
nuxl_out_q_df = calcQ(nuxl_out_df, scoreColName = 'Score', labelColName = 'Label', isXLColName = 'NuXL:isXL', addXlQ = True, ascending = False, remove_decoy=True)
print("XLs at CSM level FDR: ")
FDR_summary(nuxl_out_q_df, isXLColName="NuXL:isXL", q_val_col="class-specific_q-val")
print("XLs at PSM level FDR: ")
FDR_summary(nuxl_out_q_df, isXLColName="NuXL:isXL", q_val_col="q-val")
nuxl_out_q_df.to_csv(file_name.split('.')[0]+"_csm_q_val.csv")

print("\n --- XL crosslink peptide sequence  ----")
# just keep unmodified version of peptide e-g PEPTID(XL)E, 0.31; PEP(XL)TIDE, 0.51; PEP(XL)TIDE, 0.41 will select ==> PE(XL)PTIDE, 0.51 
# dont care protein either shared or unique peptide
nuxl_out_q_PEP_df = calc_xl_peptide_qval(nuxl_out_df, scoreColName = 'Score', labelColName = 'Label', isXLColName = 'NuXL:isXL', ascending = False, remove_decoy=True)
FDR_summary(nuxl_out_q_PEP_df, isXLColName="NuXL:isXL", q_val_col="q-val") # all are crosslink peptide just one class
nuxl_out_q_PEP_df.to_csv(file_name.split('.')[0]+"_pep_q_val.csv")

## XL proteins unique crosslink peptide sequence 
print("\n --- XL proteins unique crosslink peptide sequence ----")
# First keep unique peptides means peptide map to one protein
# Furthur keep unmodified version of peptide e-g PEPTID(XL)E, 0.31; PEP(XL)TIDE, 0.51; PEPT(XL)IDE, 0.41 will select ==> PEPTIDE, 0.51 
nuxl_out_q_PRT_df, nuxl_out_q_PRT_1 = calculate_xl_prt_qval(nuxl_out_df, score_col = "Score", seq_col = "Peptide", prot_col = 'accessions', label_col = "Label", isXLColName="NuXL:isXL", remove_decoy=True)
print("no. of proteins: ", len(nuxl_out_q_PRT_1["accessions"]))
nuxl_out_q_PRT_df.to_csv(file_name.split('.')[0]+"_prt_q_val.csv") 

"""