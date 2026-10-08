"""Optional two-layer SVR--LSTM example; needs PyTorch. Not manuscript code."""
from pathlib import Path
import argparse, copy, json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_predict
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from experiment import FEATURES, validate, ROOT

def run(path, epochs=100, seed=2026):
    import torch
    from torch import nn
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    df=pd.read_csv(path);validate(df)
    tr=df[df.split=='train'].copy();va=df[df.split=='validation'].copy();te=df[df.split=='test'].copy()
    pipe=Pipeline([('scale',StandardScaler()),('svr',SVR())]);folds=GroupKFold(5)
    search=GridSearchCV(pipe,{'svr__C':[1.,10.],'svr__gamma':['scale',.1]},cv=folds,scoring='neg_mean_absolute_error')
    search.fit(tr[FEATURES],tr.elbow_future_deg,groups=tr.subject_id)
    # Nested grouped CV: tuning and normalization fit only inner-training subjects.
    train_oof=cross_val_predict(search,tr[FEATURES],tr.elbow_future_deg,cv=folds,
        groups=tr.subject_id,params={'groups':tr.subject_id.to_numpy()})
    scale=StandardScaler().fit(tr[FEATURES]);target_scale=StandardScaler().fit(tr[['elbow_future_deg']].to_numpy())
    data=[]
    for d,pred in [(tr,train_oof),(va,search.predict(va[FEATURES])),(te,search.predict(te[FEATURES]))]:
        d=d.copy();aug=np.c_[scale.transform(d[FEATURES]),target_scale.transform(np.array(pred).reshape(-1,1))]
        d['_row']=np.arange(len(d));xs=[];ys=[];ids=[]
        for (sid,trial),g in d.groupby(['subject_id','trial_id']):
            g=g.sort_values('window_index');ix=g._row.to_numpy()
            for end in range(4,len(ix)):
                if not np.all(np.diff(g.window_index.to_numpy()[end-4:end+1])==1):continue
                xs.append(aug[ix[end-4:end+1]]);ys.append(target_scale.transform([[g.elbow_future_deg.iloc[end]]])[0]);ids.append(sid)
        if not xs:raise ValueError('No length-5 contiguous sequences')
        data.append((torch.tensor(np.array(xs),dtype=torch.float32),torch.tensor(np.array(ys),dtype=torch.float32),ids))
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.lstm=nn.LSTM(5,64,num_layers=2,dropout=.2,batch_first=True);self.head=nn.Linear(64,1)
        def forward(self,x):return self.head(self.lstm(x)[0][:,-1])
    model=Model();opt=torch.optim.Adam(model.parameters(),lr=.001);loss=nn.MSELoss();best=float('inf');state=None;wait=0
    history=[];rng=np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train();order=rng.permutation(len(data[0][0]))
        for start in range(0,len(order),64):
            ix=order[start:start+64];opt.zero_grad();l=loss(model(data[0][0][ix]),data[0][1][ix]);l.backward();opt.step()
        model.eval()
        with torch.no_grad():vl=float(loss(model(data[1][0]),data[1][1]))
        history.append([epoch+1,vl])
        if vl<best:best=vl;state=copy.deepcopy(model.state_dict());wait=0
        else:wait+=1
        if wait>=10:break
    model.load_state_dict(state)
    with torch.no_grad():pred=target_scale.inverse_transform(model(data[2][0]).numpy()).ravel()
    truth=target_scale.inverse_transform(data[2][1].numpy()).ravel();out=ROOT/'results/lstm';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'subject_id':data[2][2],'truth_deg':truth,'prediction_deg':pred}).to_csv(out/'test_predictions.csv',index=False)
    pd.DataFrame(history,columns=['epoch','validation_mse_standardized']).to_csv(out/'training_history.csv',index=False)
    (out/'metrics.json').write_text(json.dumps({'provenance':sorted(df.provenance.unique().tolist()),'mae_deg':float(mean_absolute_error(truth,pred)),
      'rmse_deg':float(np.sqrt(mean_squared_error(truth,pred))),'epochs':len(history),'seed':seed,'scope':'New implementation; not original experiment'},indent=2))
    torch.save(model.state_dict(),out/'model.pt')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data/demo_features.csv');p.add_argument('--epochs',type=int,default=100);a=p.parse_args();run(a.data,a.epochs)
