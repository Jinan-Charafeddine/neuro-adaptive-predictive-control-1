"""Executable feature benchmark and reduced-order controller demonstration.
This is newly implemented example code, not the authors' original experiment.
"""
from pathlib import Path
import argparse, hashlib, json, platform, sys
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_predict
from sklearn.svm import SVR, SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['biceps_rms', 'triceps_rms', 'ant_deltoid_rms', 'post_deltoid_rms']
CLASSES = ['extension', 'flexion', 'hold']


def generate(seed=2026):
    """27 artificial IDs, not 27 patients; directly generated window features."""
    rng = np.random.default_rng(seed)
    people, rows = [], []
    for n in range(27):
        sid = f'DEMO_{n+1:03d}'
        split = 'train' if n < 19 else ('validation' if n < 23 else 'test')
        people.append([sid, split, 'demo', 'not_applicable'])
        scale = rng.uniform(.8, 1.2, 4)
        for trial in range(3):
            t = np.arange(60) * .1
            phase = 2*np.pi*t/6
            elbow = 60 + 25*np.sin(phase)
            shoulder = 30 + 12*np.sin(phase + .2)
            v = np.cos(phase)
            labels = np.where(v > .25, 'flexion', np.where(v < -.25, 'extension', 'hold'))
            noise = rng.normal(0, .035, (60,4))
            features = np.stack([.25+.45*np.maximum(v,0), .25+.45*np.maximum(-v,0),
                                 .25+.25*np.maximum(np.cos(phase+.2),0),
                                 .25+.25*np.maximum(-np.cos(phase+.2),0)], axis=1)
            features = np.clip(features*scale + noise, .01, 1)
            for i in range(59):
                rows.append([sid, f'T{trial+1:02d}', i, float(t[i]), split, 'demo',
                             *features[i], float(elbow[i]), float(shoulder[i]), labels[i],
                             float(elbow[i+1]), float(shoulder[i+1])])
    cols = ['subject_id','trial_id','window_index','time_s','split','provenance', *FEATURES,
            'elbow_deg','shoulder_deg','intention','elbow_future_deg','shoulder_future_deg']
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(ROOT/'data/demo_features.csv', index=False)
    pd.DataFrame(people, columns=['subject_id','split','provenance','diagnosis']).to_csv(ROOT/'data/demo_subjects.csv',index=False)
    pd.DataFrame(columns=cols).to_csv(ROOT/'data/real_features_template.csv',index=False)
    pd.DataFrame(columns=['subject_id','split','provenance','diagnosis']).to_csv(ROOT/'data/real_subjects_template.csv',index=False)
    return df


def validate(df):
    required = set(['subject_id','trial_id','window_index','time_s','split','provenance',*FEATURES,
                    'intention','elbow_future_deg','shoulder_future_deg'])
    if required-set(df): raise ValueError(f'Missing columns: {sorted(required-set(df))}')
    if df.empty: raise ValueError('Dataset is empty; populate the real-data template first.')
    if not set(df.split).issubset({'train','validation','test'}): raise ValueError('Invalid split labels')
    if df.groupby('subject_id').split.nunique().max()!=1: raise ValueError('Participant leakage across splits')
    if df.duplicated(['subject_id','trial_id','window_index']).any(): raise ValueError('Duplicate windows')
    if not set(df.intention).issubset(CLASSES): raise ValueError('Invalid intention labels')
    if not np.isfinite(df[FEATURES+['elbow_future_deg','shoulder_future_deg']].to_numpy(float)).all():
        raise ValueError('Nonfinite numeric values')
    for split in ['train','validation','test']:
        if df[df.split==split].empty: raise ValueError(f'Empty {split} subset')
    if df[df.split=='train'].subject_id.nunique()<5: raise ValueError('Five grouped CV folds require >=5 training subjects')
    return True


def balanced_indices(y, rng, cap=60):
    groups=[np.flatnonzero(y==c) for c in CLASSES]
    size=min(cap,*(len(g) for g in groups))
    if not size: raise ValueError('A training intention class is absent')
    return np.concatenate([rng.choice(g,size,replace=False) for g in groups])


def statevectors(x):
    """Four-qubit, two-repeat H/diagonal-Z/ZZ feature map, computed in NumPy.
    Defined explicitly here; not asserted identical to any Qiskit version.
    """
    x=np.asarray(x,float)
    if x.ndim!=2 or x.shape[1]!=4: raise ValueError('Requires four features')
    z=np.array([[1 if ((b>>q)&1)==0 else -1 for q in range(4)] for b in range(16)])
    angles=x@z.T
    for q in range(3): angles += ((np.pi-x[:,q])*(np.pi-x[:,q+1]))[:,None]*(z[:,q]*z[:,q+1])
    phase=np.exp(-1j*angles)
    h=np.array([[1,1],[1,-1]])/np.sqrt(2)
    H=h
    for _ in range(3): H=np.kron(H,h)
    psi=np.zeros((len(x),16),complex); psi[:,0]=1
    for _ in range(2): psi=(psi@H.T)*phase
    return psi


def kernel(a,b): return np.abs(statevectors(a).conj()@statevectors(b).T)**2


def bootstrap_subject_accuracy(pred, seed=2026):
    """Subject-cluster bootstrap; descriptive and unstable with very few IDs."""
    rng=np.random.default_rng(seed)
    scores=pred.groupby('subject_id').apply(lambda d: float(np.mean(d.truth==d.prediction)),include_groups=False)
    means=[rng.choice(scores.to_numpy(),len(scores),replace=True).mean() for _ in range(1000)]
    return {'subject_mean_accuracy':float(scores.mean()),'subject_bootstrap_95pct':np.percentile(means,[2.5,97.5]).tolist(),
            'test_subjects':len(scores),'interpretation':'descriptive; demonstration unless real input supplied'}


def train(df, out, seed=2026):
    validate(df); rng=np.random.default_rng(seed)
    tr=df[df.split=='train']; va=df[df.split=='validation']; te=df[df.split=='test']
    metrics={'provenance':sorted(df.provenance.unique().tolist()), 'seed':seed, 'classifier':{},'regression':{}}
    df.groupby(['split','intention']).size().rename('windows').reset_index().to_csv(out/'class_counts.csv',index=False)
    cv=GroupKFold(5)
    for target in ['elbow_future_deg','shoulder_future_deg']:
        pipe=Pipeline([('scale',StandardScaler()),('svr',SVR(kernel='rbf'))])
        search=GridSearchCV(pipe,{'svr__C':[1.,10.],'svr__gamma':['scale',.1]},cv=cv,scoring='neg_mean_absolute_error',n_jobs=1)
        search.fit(tr[FEATURES],tr[target],groups=tr.subject_id)
        p=search.predict(te[FEATURES])
        metrics['regression'][target]={'model':'SVR','parameters':search.best_params_,
          'mae_deg':float(mean_absolute_error(te[target],p)),
          'rmse_deg':float(np.sqrt(mean_squared_error(te[target],p))),
          'validation_mae_deg':float(mean_absolute_error(va[target],search.predict(va[FEATURES])))}
        pd.DataFrame({'subject_id':te.subject_id,'trial_id':te.trial_id,'truth_deg':te[target],'prediction_deg':p}).to_csv(out/f'svr_{target}.csv',index=False)
    # Both classifiers use the identical balanced training subset; validation chooses C.
    idx=balanced_indices(tr.intention.to_numpy(),rng)
    bt=tr.iloc[idx]
    standard=StandardScaler().fit(bt[FEATURES])
    xt=standard.transform(bt[FEATURES]); xv=standard.transform(va[FEATURES]); xe=standard.transform(te[FEATURES])
    # Separate training-only angular scaling for the feature map.
    angular=MinMaxScaler(feature_range=(0,np.pi)).fit(xt)
    qt=angular.transform(xt); qv=angular.transform(xv); qe=angular.transform(xe)
    kernels=[kernel(qt,qt),kernel(qv,qt),kernel(qe,qt)]
    for name in ['classical_svm','statevector_kernel_svm']:
        best=None
        for C in [.1,1.,10.]:
            model=SVC(C=C,kernel='rbf' if name=='classical_svm' else 'precomputed',gamma='scale')
            model.fit(xt if name=='classical_svm' else kernels[0],bt.intention)
            score=accuracy_score(va.intention,model.predict(xv if name=='classical_svm' else kernels[1]))
            if best is None or score>best[0]:best=(score,C,model)
        p=best[2].predict(xe if name=='classical_svm' else kernels[2])
        pred=pd.DataFrame({'subject_id':te.subject_id,'trial_id':te.trial_id,'truth':te.intention,'prediction':p})
        pred.to_csv(out/f'{name}_predictions.csv',index=False)
        cm=confusion_matrix(te.intention,p,labels=CLASSES)
        pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(out/f'{name}_confusion_matrix.csv')
        metrics['classifier'][name]={'accuracy':float(accuracy_score(te.intention,p)), 'selected_C':best[1],
           'validation_accuracy':float(best[0]),'balanced_training_samples':len(bt),
           'per_class':classification_report(te.intention,p,labels=CLASSES,output_dict=True,zero_division=0),
           **bootstrap_subject_accuracy(pred,seed)}
        fig,ax=plt.subplots(figsize=(5,4));ax.imshow(cm,cmap='Blues')
        ax.set(xticks=range(3),yticks=range(3),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted',ylabel='True',title=f'{name}\nDemo data, not manuscript results')
        for i in range(3):
            for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center')
        fig.tight_layout();fig.savefig(out/f'{name}_confusion_matrix.png',dpi=150);plt.close(fig)
    return metrics


def trap(x,a,b,c,d):
    x=np.asarray(x); left=np.ones_like(x) if a==b else (x-a)/(b-a)
    right=np.ones_like(x) if c==d else (d-x)/(d-c)
    return np.clip(np.minimum(left,right),0,1)*((x>=a)&(x<=d))


def memberships(x,scale=1):
    return np.array([trap(x,0,0,.25*scale,.5*scale),trap(x,.25*scale,.5*scale,.5*scale,.75*scale),trap(x,.5*scale,.75*scale,scale,scale)])


def fuzzy(f1,f2):
    universe=np.linspace(0,2,201); outputs=memberships(universe,2)
    rules=np.array([[0,0,1],[0,1,2],[1,2,2]])
    a=memberships(np.clip(f1,0,1));b=memberships(np.clip(f2,0,1)); agg=np.zeros_like(universe)
    for i in range(3):
        for j in range(3):agg=np.maximum(agg,np.minimum(min(a[i],b[j]),outputs[rules[i,j]]))
    return float(np.sum(universe*agg)/np.sum(agg))


def cci(a,b): return 2*np.minimum(a,b)/(a+b+1e-8)


def controller_demo(out,seed=2026):
    """Illustrative single-joint PD surrogate, NOT OpenSim or the manuscript admittance controller.
    Muscle proxies are derived from residual user-torque demand by construction.
    """
    records=[]; traces=[]
    dt=.01;t=np.arange(0,4,dt);ref=np.deg2rad(60+20*np.sin(2*np.pi*t/4));rv=np.gradient(ref,dt)
    cfg={'provenance':'reduced_order_demo','backend':'single-joint Euler PD surrogate; not OpenSim',
         'dt_s':dt,'inertia_kg_m2':.08,'damping_Nm_s_rad':.15,'torque_limit_Nm':3.,
         'joint_limits_rad':[0,np.deg2rad(130)],'seeds':list(range(seed,seed+20)),
         'not_manuscript_parameters':True,'fatigue_force_scale_end':.65,'spastic_burst_sigma_s':.06}
    for run in range(20):
        for condition in ['nominal','noise','burst_fatigue']:
            rng=np.random.default_rng(seed+run)
            noise=rng.normal(0,.01 if condition=='noise' else .002,len(t))
            burst=.5*np.exp(-.5*((t-rng.uniform(1,3))/.06)**2) if condition=='burst_fatigue' else np.zeros(len(t))
            fatigue=np.linspace(1,.65,len(t)) if condition=='burst_fatigue' else np.ones(len(t))
            for controller in ['C1_no_assistance','C2_fixed_PD','C3_NMI_adaptive_PD_demo']:
                q=ref[0];v=0.;qs=[];us=[];eff=[];cc=[]; violations=0; saturations=0
                for i in range(len(t)):
                    observed=q+noise[i];err=ref[i]-observed
                    base=2*err+.4*(rv[i]-v)+.2*np.sin(q)
                    ag=max(base,0)+.12+burst[i]; ant=max(-base,0)+.12+burst[i]
                    ci=float(cci(ag,ant));nmi=float(np.clip(.8*ci+.2*abs(err)/np.deg2rad(30),0,1))
                    if controller.startswith('C1'):raw=0.
                    elif controller.startswith('C2'):raw=3*err+.3*(rv[i]-v)
                    else:raw=(4-2*nmi)*err+(.25+.55*nmi)*(rv[i]-v)
                    u=float(np.clip(raw,-cfg['torque_limit_Nm'],cfg['torque_limit_Nm']));saturations+=int(raw!=u)
                    residual=base-u
                    # Effort/CCI proxies do not represent physiological measurements.
                    agon=max(residual,0)+.12+burst[i];antag=max(-residual,0)+.12+burst[i]
                    user=fatigue[i]*residual
                    acceleration=(user+u-.15*v-.2*np.sin(q)+burst[i])/.08
                    v+=acceleration*dt; q+=v*dt
                    outside=q<0 or q>cfg['joint_limits_rad'][1]; violations+=int(outside)
                    if outside:q=float(np.clip(q,0,cfg['joint_limits_rad'][1]));v=0.
                    qs.append(q);us.append(u);eff.append(np.sqrt((agon**2+antag**2)/2));cc.append(float(cci(agon,antag)))
                error=np.rad2deg(np.array(qs)-ref)
                records.append([run,condition,controller,float(np.mean(abs(error))),float(np.sqrt(np.mean(error**2))),
                                float(np.max(abs(error))),float(np.mean(eff)),float(np.mean(cc)),violations,saturations])
                if run==0 and condition=='nominal':
                    traces.extend([[float(t[i]),controller,float(np.rad2deg(ref[i])),float(np.rad2deg(qs[i])),us[i]] for i in range(len(t))])
    columns=['run','condition','controller','mae_deg','rmse_deg','max_abs_error_deg','activation_effort_proxy','cci_proxy','preclip_joint_limit_violations','torque_saturations']
    runs=pd.DataFrame(records,columns=columns);runs.to_csv(out/'controller_demo_runs.csv',index=False)
    runs.groupby(['condition','controller'])[columns[3:]].agg(['mean','std']).to_csv(out/'controller_demo_summary.csv')
    trace=pd.DataFrame(traces,columns=['time_s','controller','reference_deg','joint_deg','assist_torque_Nm']);trace.to_csv(out/'controller_demo_trace.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4));ax.plot(t,np.rad2deg(ref),'k--',label='Reference')
    for name,d in trace.groupby('controller'):ax.plot(d.time_s,d.joint_deg,label=name)
    ax.set(xlabel='Time (s)',ylabel='Joint angle (degrees)',title='Reduced-order controller demonstration (not OpenSim)');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'controller_demo_tracking.png',dpi=150);plt.close(fig)
    (out/'controller_demo_config.json').write_text(json.dumps(cfg,indent=2))
    rng=np.random.default_rng(seed);loads=rng.uniform(0,2,120)
    a=np.clip(loads/2+rng.normal(0,.07,len(loads)),0,1);b=np.clip(loads/2+rng.normal(0,.07,len(loads)),0,1)
    est=np.array([fuzzy(x,y) for x,y in zip(a,b)])
    pd.DataFrame({'provenance':'demo','known_load_kg':loads,'f1':a,'f2':b,'estimated_load_kg':est,'absolute_error_kg':abs(est-loads)}).to_csv(out/'fuzzy_load_demo.csv',index=False)
    return {'mae_kg':float(np.mean(abs(est-loads))),'nmae_percent':float(np.mean(abs(est-loads))/2*100),'provenance':'demo'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--output',type=Path,default=ROOT/'results');p.add_argument('--generate-only',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.data) if a.data else generate(a.seed)
    if a.generate_only:return
    metrics=train(df,a.output,a.seed)
    # Controller demonstration is even when real offline features are supplied.
    metrics['fuzzy_demo']=controller_demo(a.output,a.seed)
    metrics['scope']='New demonstration implementation; not reproduction of published manuscript numbers or OpenSim model.'
    metrics['environment']={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn_version}
    source=a.data or ROOT/'data/demo_features.csv'
    metrics['dataset_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    (a.output/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps({'data':str(source),'results':str(a.output),'provenance':metrics['provenance'],'scope':metrics['scope']},indent=2))

if __name__=='__main__':main()
