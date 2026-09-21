"""Add frozen Chronos forecasts to existing benchmark without refitting others."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline
from sklearn.metrics import mean_absolute_error,r2_score
from src.data import load_panel

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='amazon/chronos-bolt-tiny');args=ap.parse_args()
 c=json.loads(Path('config.json').read_text(encoding='utf-8'));p,ids,q=load_panel('data/consumption.parquet',c);out=Path('results')
 torch.set_num_threads(2)
 kw={} if Path(args.model).exists() else {'revision':c['chronos_revision']}
 model=BaseChronosPipeline.from_pretrained(args.model,device_map='cpu',torch_dtype=torch.float32,**kw)
 assert np.isclose(model.quantiles[4],.5), 'Expected median quantile at index 4'
 dates=pd.to_datetime(p.columns);rows=[]
 for split in ['validation','test']:
  for h in c['horizons']:
   origins=c['validation_origins'] if split=='validation' else c[f'test_origins_{h}']
   for origin in origins:
    ys=p.loc[ids].iloc[:,:origin].T.ffill().T.to_numpy(float)
    pred=np.maximum(0,model.predict(torch.tensor(ys,dtype=torch.float32),prediction_length=h).cpu().numpy()[:,4,:])
    for i,id_ in enumerate(ids):
     for step in range(1,h+1):
      actual=p.loc[id_].iloc[origin+step-1]
      if pd.notna(actual):rows.append(dict(split=split,horizon=h,step=step,origin=str(dates[origin-1].date()),date=str(dates[origin+step-1].date()),territory_id=id_,model='chronos',actual=float(actual),prediction=float(pred[i,step-1])))
    print(split,h,origin,flush=True)
 old=pd.read_csv(out/'predictions.csv');df=pd.concat([old[old.model!='chronos'],pd.DataFrame(rows)],ignore_index=True);df.to_csv(out/'predictions.csv',index=False)
 metrics=[]
 for (split,h,name),g in df.groupby(['split','horizon','model']):metrics.append(dict(split=split,horizon=h,model=name,n=len(g),MAE=mean_absolute_error(g.actual,g.prediction),R2=r2_score(g.actual,g.prediction)))
 m=pd.DataFrame(metrics);m.to_csv(out/'metrics.csv',index=False)
 selections={};boot=[]
 for h in c['horizons']:
  selected=m[(m.split=='validation')&(m.horizon==h)].sort_values('MAE').iloc[0].model;selections[str(h)]=selected
  test=df[(df.split=='test')&(df.horizon==h)];k=['territory_id','origin','date','step']
  z=test[test.model==selected].merge(test[test.model=='prophet'],on=k,suffixes=('_a','_b'));z['gain']=abs(z.actual_b-z.prediction_b)-abs(z.actual_a-z.prediction_a)
  grouped=z.groupby('territory_id').gain.agg(['sum','count']).to_numpy();rng=np.random.default_rng(c['seed'])
  samples=[grouped[rng.integers(0,len(grouped),len(grouped))].sum(axis=0) for _ in range(c['bootstrap_repeats'])];means=[a/b for a,b in samples]
  boot.append({'horizon':h,'selected':selected,'MAE_gain_rub':z.gain.mean(),'ci_low':np.quantile(means,.025),'ci_high':np.quantile(means,.975)})
  if selected=='chronos':
   ys=p.loc[ids].T.ffill().T.to_numpy(float);pred=np.maximum(0,model.predict(torch.tensor(ys,dtype=torch.float32),prediction_length=h).cpu().numpy()[:,4,:])
   pd.DataFrame([{'territory_id':id_,'date':str((dates[-1]+pd.DateOffset(months=j+1)).date()),'prediction':float(pred[i,j]),'model':selected} for i,id_ in enumerate(ids) for j in range(h)]).to_csv(out/f'forecast_h{h}.csv',index=False)
 (out/'selection.json').write_text(json.dumps(selections,indent=2));pd.DataFrame(boot).to_csv(out/'bootstrap.csv',index=False)
 print(m.to_string(index=False))

if __name__=='__main__':main()
