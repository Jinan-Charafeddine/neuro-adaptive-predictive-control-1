"""Example causal raw-sEMG preprocessing for an authorized synchronized CSV.
Expected rows: 1000-Hz sEMG, joint angles and labels aligned to that grid.
Not asserted to match the original acquisition/processing implementation.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfilt
from experiment import FEATURES
RAW=['biceps_uV','triceps_uV','ant_deltoid_uV','post_deltoid_uV']

def extract(raw_path,output):
    raw=pd.read_csv(raw_path)
    required=['subject_id','trial_id','split','time_s','elbow_deg','shoulder_deg','intention']+RAW
    if set(required)-set(raw):raise ValueError(f'Required columns: {required}')
    if raw.groupby('subject_id').split.nunique().max()!=1:raise ValueError('Subject crosses splits')
    if raw[required].isna().any().any():raise ValueError('Missing raw values or labels')
    band=butter(4,[20,450],btype='bandpass',fs=1000,output='sos')
    smooth=butter(4,5,btype='lowpass',fs=1000,output='sos')
    blocks=[]
    for (sid,trial),g in raw.groupby(['subject_id','trial_id']):
        g=g.sort_values('time_s').copy()
        if not np.allclose(np.diff(g.time_s),.001,atol=1e-5):raise ValueError('Expected uniform 1000-Hz synchronized rows')
        signals=g[RAW].to_numpy(float)
        filtered=sosfilt(band,signals,axis=0)
        envelope=np.maximum(sosfilt(smooth,abs(filtered),axis=0),0)
        blocks.append((sid,trial,g,envelope))
    training=[x[3] for x in blocks if x[2].split.iloc[0]=='train']
    if not training:raise ValueError('No training recordings')
    scale=np.maximum(np.max(np.vstack(training),axis=0),1e-8)
    rows=[]
    for sid,trial,g,envelope in blocks:
        for idx,start in enumerate(range(0,len(g)-299,100)):
            end=start+199;future=end+100
            f=np.sqrt(np.mean((envelope[start:end+1]/scale)**2,axis=0))
            rows.append([sid,trial,idx,float(g.time_s.iloc[end]),g.split.iloc[end],
             'clinical_recording_features',*f,float(g.elbow_deg.iloc[end]),float(g.shoulder_deg.iloc[end]),
             g.intention.iloc[end],float(g.elbow_deg.iloc[future]),float(g.shoulder_deg.iloc[future])])
    cols=['subject_id','trial_id','window_index','time_s','split','provenance',*FEATURES,
          'elbow_deg','shoulder_deg','intention','elbow_future_deg','shoulder_future_deg']
    output.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows,columns=cols).to_csv(output,index=False)
    output.with_suffix('.preprocessing.json').write_text(json.dumps({'sampling_hz':1000,'window_ms':200,'stride_ms':100,
      'prediction_horizon_ms':100,'causal_filter':True,'training_peak_scales_uV':scale.tolist(),
      'filter_transients':'not discarded; assess and specify warm-up before research use',
      'inner_CV_note':'For strict fold-specific peak scaling, rerun normalization within each training fold; this outer-training export alone does not establish that.'},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('--output',type=Path,default=Path('data/private/features.csv'));a=p.parse_args();extract(a.raw,a.output)
