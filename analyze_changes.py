"""Detectors on synthetic truth; real alerts and future residual proxy risk."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,f1_score,precision_score,recall_score
from src.data import load_panel,news_features
from src.models import seasonal_scaled
from src.detection import detect,synthetic_benchmark,evaluate_detector

def main():
 c=json.loads(Path('config.json').read_text(encoding='utf-8'));out=Path('results')
 p,ids,q=load_panel('data/consumption.parquet',c);news=pd.read_csv('data/news.csv');dates=pd.to_datetime(p.columns)
 dc=c['detector'];rc=c['risk'];val=synthetic_benchmark(dc['synthetic_validation_seed'],dc['synthetic_cases']);test=synthetic_benchmark(dc['synthetic_test_seed'],dc['synthetic_cases']);grid=[];scores=[]
 for method in ['threshold','cusum','ewma']:
  candidates=[]
  for th in dc['threshold_grid']:
   v=evaluate_detector(val,method,th);row={'method':method,'threshold':th,**v};grid.append(row);candidates.append(row)
  best=max(candidates,key=lambda x:x['F1']);scores.append({'split':'test','method':method,'threshold':best['threshold'],**evaluate_detector(test,method,best['threshold'])})
 pd.DataFrame(grid).to_csv(out/'detector_validation.csv',index=False);pd.DataFrame(scores).to_csv(out/'detector_test.csv',index=False)
 winner=max(grid,key=lambda x:x['F1']);(out/'detector_selection.json').write_text(json.dumps(winner,indent=2))
 residuals=[]
 for id_,row in p.iterrows():
  y=row.to_numpy(float)
  for t in range(12,24):
   if np.isfinite(y[:t+1]).all():
    pred=seasonal_scaled(y[:t],1)[0];residuals.append({'territory_id':id_,'date':str(dates[t].date()),'t':t-12,'residual':(y[t]-pred)/max(pred,1),'actual':y[t],'prediction':pred})
 r=pd.DataFrame(residuals)
 train=r[r.t<6].residual.to_numpy();center=float(np.median(train));scale=float(max(1.4826*np.median(abs(train-center)),.01))
 alerts=[]
 for id_,g in r[r.territory_id.isin(ids)].groupby('territory_id'):
  g=g.sort_values('t');monitor=g[g.t>=6];z=(monitor.residual.to_numpy()-center)/scale
  for s in scores:
   hits,values=detect(z,s['method'],s['threshold'],cooldown=dc['real_cooldown'])
   for i in hits:alerts.append({'territory_id':int(id_),'date':monitor.iloc[i].date,'method':s['method'],'score':values[i],'residual_pct':100*monitor.iloc[i].residual})
 pd.DataFrame(alerts,columns=['territory_id','date','method','score','residual_pct']).to_csv(out/'real_alerts.csv',index=False)
 r.to_csv(out/'residuals.csv',index=False)
 (out/'residual_calibration.json').write_text(json.dumps({'median':center,'mad_scale':scale,'calibration':'2024-01 to 2024-06','monitoring':'2024-07 to 2024-12'},indent=2))
 # Future label is a PROXY: two next residuals >8% in magnitude, same sign.
 samples=[]
 for id_,g in r.groupby('territory_id'):
  g=g.set_index('t')
  for t in [1,2,3,5,6,8,9]:
   if not all(k in g.index for k in [t-1,t,t+1,t+2]):continue
   a,b=g.loc[t+1,'residual'],g.loc[t+2,'residual'];label=int(abs(a)>rc['residual_threshold'] and abs(b)>rc['residual_threshold'] and a*b>0)
   now=g.loc[t,'residual'];prev=g.loc[t-1,'residual'];nf=news_features(news,pd.Timestamp(g.loc[t,'date'])+pd.offsets.MonthEnd(0))
   samples.append({'territory_id':int(id_),'t':t,'label':label,'residual':now,'previous':prev,'absolute':abs(now),'change':now-prev,'rate':nf[0],'news_delta':nf[1],'news_count':nf[2]})
 d=pd.DataFrame(samples);tr=d[d.t.isin([1,2,3])];va=d[d.t.isin([5,6])&d.territory_id.isin(ids)];te=d[d.t.isin([8,9])&d.territory_id.isin(ids)]
 risk=[];preds=[]
 base=['residual','previous','absolute','change']
 for name in ['constant','risk','risk_news']:
  cols=base+(['rate','news_delta','news_count'] if name=='risk_news' else [])
  if name=='constant':pv=np.repeat(tr.label.mean(),len(va));pt=np.repeat(tr.label.mean(),len(te))
  elif tr.label.nunique()<2:raise ValueError('Risk training requires both classes')
  else:
   model=make_pipeline(StandardScaler(),LogisticRegression(C=rc['logistic_C'],max_iter=1000,random_state=c['seed']));model.fit(tr[cols],tr.label);pv=model.predict_proba(va[cols])[:,1];pt=model.predict_proba(te[cols])[:,1]
  # If validation has no events, F1 cannot identify a useful threshold.
  # Use a conservative fixed 0.5 and disclose insufficient validation labels.
  threshold=max(np.arange(.05,.96,.05),key=lambda x:f1_score(va.label,pv>=x,zero_division=0)) if va.label.nunique()>1 else rc['fallback_probability_threshold']
  for split,dd,prob in [('validation',va,pv),('test',te,pt)]:
   risk.append({'model':name,'split':split,'n':len(dd),'events':int(dd.label.sum()),'threshold':threshold,'PR_AUC_AP':average_precision_score(dd.label,prob) if dd.label.sum()>0 else None,'ROC_AUC':roc_auc_score(dd.label,prob) if dd.label.nunique()>1 else None,'Brier':brier_score_loss(dd.label,prob),'F1':f1_score(dd.label,prob>=threshold,zero_division=0),'precision':precision_score(dd.label,prob>=threshold,zero_division=0),'recall':recall_score(dd.label,prob>=threshold,zero_division=0)})
   for (_,rr),pr in zip(dd.iterrows(),prob):preds.append({'split':split,'model':name,'territory_id':int(rr.territory_id),'origin_t':int(rr.t),'label':int(rr.label),'probability':float(pr)})
 pd.DataFrame(risk).to_csv(out/'risk_metrics.csv',index=False);pd.DataFrame(preds).to_csv(out/'risk_predictions.csv',index=False)
 print(pd.DataFrame(scores).to_string(index=False));print(pd.DataFrame(risk).to_string(index=False))

if __name__=='__main__':main()
