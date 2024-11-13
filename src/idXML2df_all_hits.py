import pandas as pd
from pyopenms import IdXMLFile

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

def readAndProcessIdXMLAllHits(input_file):
  """
  convert the (.idXML) format identification file to dataframe
  Get all hits (ranks) per peptide from idXML file.
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
      #if psm_index > top:
        #print("break!!!")
       #break
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
      #break; # parse only first hit
  
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
