import os
import pandas as pd
import numpy as np
from nilearn import datasets, maskers, connectome
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

DATASET_ROOT = r'D:\ADHD200\RawDataBIDS'
OUTPUT_FILE = r'D:\ADHD200\RawDataBIDS\adhd_processed_data.npy'
N_JOBS = -1 
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7)
masker = maskers.NiftiLabelsMasker(
    labels_img=atlas.maps,
    standardize="zscore_sample",
    verbose=0
)
correlation_measure = connectome.ConnectivityMeasure(kind='correlation')

def process_subject(row_data):
    site_path, participant_id, dx_raw, gender_raw, age_raw, iq_raw = row_data
    
    try:
       
        sub_id = str(participant_id)
        formatted_sub_id = f"sub-{sub_id.zfill(7)}" if not sub_id.startswith('sub-') else sub_id
        
        
        subject_dir = os.path.join(site_path, formatted_sub_id)
        fmri_path = None
        
        if os.path.exists(subject_dir):
            for root, _, files in os.walk(subject_dir):
                for file in files:
                    if file.endswith('bold.nii.gz'):
                        fmri_path = os.path.join(root, file)
                        break
                if fmri_path: break
        
        if not fmri_path: return None

       
        label = 0
        val = str(dx_raw).lower()
        if any(x in val for x in ["typically", "control", "td", "0"]):
            label = 0
        elif any(x in val for x in ["adhd", "1", "2", "3"]):
            label = 1
        else:
            return None 

        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            time_series = masker.fit_transform(fmri_path)
            
            matrix = correlation_measure.fit_transform([time_series])[0]

        
        gender = 0 if str(gender_raw).lower() == 'male' else 1
        
        try: age = float(age_raw)
        except: age = 0.0
        
        try: iq = float(iq_raw)
        except: iq = 100.0
        if np.isnan(iq): iq = 100.0

        return {
            'matrix': matrix, 
            'pheno': [age, gender, iq], 
            'label': label
        }

    except Exception:
        return None


if __name__ == "__main__":
    sites = ['NeuroIMAGE', 'KKI', 'Pittsburgh', 'Peking_1', 'OHSU', 'WashU', 'Brown']
    tasks = []
    
    print("Gathering tasks from Google Drive...")
    for site in sites:
        site_path = os.path.join(DATASET_ROOT, site)
        pheno_file = os.path.join(site_path, 'participants.tsv')
        
        if not os.path.exists(pheno_file): continue
        
        try:
            df = pd.read_csv(pheno_file, sep='\t', encoding_errors='replace')
            df.columns = [c.lower() for c in df.columns]
            
            target_col = next((c for c in ['dx', 'diagnosis', 'adhd_status', 'group'] if c in df.columns), None)
            iq_col = next((c for c in ['verbal_iq', 'iq'] if c in df.columns), None)
            
            if not target_col: continue

            for _, row in df.iterrows():
                iq_val = row[iq_col] if iq_col else np.nan
                tasks.append((
                    site_path, 
                    row['participant_id'], 
                    row[target_col], 
                    row['gender'], 
                    row['age'], 
                    iq_val
                ))
                
        except Exception as e:
            print(f"Error preparing {site}: {e}")

    print(f"Found {len(tasks)} scans.")
    print(f"Starting Parallel Processing (Direct Drive Access)...")
    
    results = Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(process_subject)(t) for t in tqdm(tasks, unit="scan")
    )

    valid_results = [r for r in results if r is not None]
    print(f"\nProcessing Complete!")
    print(f"Successfully processed: {len(valid_results)} subjects") 
    np.save(OUTPUT_FILE, valid_results)
    print(f"Saved to {OUTPUT_FILE}")