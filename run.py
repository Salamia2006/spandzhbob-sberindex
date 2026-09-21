"""Run from the repository directory: python run.py [--chronos-cache PATH]."""
import argparse,json,logging,os,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,r2_score
from src.data import load_panel
from src.models import pooled_predict,seasonal_scaled,prophet_predict

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config.json'); ap.add_argument('--chronos-cache'); args=ap.parse_args()
    c=json.loads(Path(args.config).read_text(encoding='utf-8')); out=Path('results');out.mkdir(exist_ok=True)
    p,ids,q=load_panel('data/consumption.parquet',c)
    news=pd.read_csv('data/news.csv');dates=pd.to_datetime(p.columns)
    (out/'data_quality.json').write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[];timings=[];cache={}
    logging.getLogger('cmdstanpy').disabled=True
    chronos=None
    if args.chronos_cache:
        import torch
        from chronos import BaseChronosPipeline
        torch.set_num_threads(2)
        chronos=BaseChronosPipeline.from_pretrained(args.chronos_cache,device_map='cpu',torch_dtype=torch.float32,local_files_only=True)
    models=['last','seasonal','seasonal_scaled','ridge','boosting','boosting_news','prophet']
    if chronos: models.append('chronos')
    for split in ['validation','test']:
      for horizon in c['horizons']:
        origins=c['validation_origins'] if split=='validation' else c[f'test_origins_{horizon}']
        for origin in origins:
          for name in models:
            key=(origin,horizon,name); started=time.perf_counter()
            if key in cache: pred=cache[key]
            elif name in ['ridge','boosting','boosting_news']:
                pred=pooled_predict(p,origin,ids,horizon,'ridge' if name=='ridge' else 'boosting',c,news if name=='boosting_news' else None)
            else:
                ys=p.loc[ids].iloc[:,:origin].T.ffill().T.to_numpy(float)
                if name=='chronos':
                    v=chronos.predict(torch.tensor(ys,dtype=torch.float32),prediction_length=horizon).cpu().numpy()
                    pred=np.maximum(0,v[:,4,:]) # median of 9 quantiles .1,...,.9
                else:
                    pred=np.array([np.repeat(y[-1],horizon) if name=='last' else y[-12:-12+horizon] if name=='seasonal' else seasonal_scaled(y,horizon) if name=='seasonal_scaled' else prophet_predict(y,dates[:origin],horizon,c) for y in ys])
            cache[key]=pred
            timings.append({'split':split,'horizon':horizon,'origin':origin,'model':name,'seconds':time.perf_counter()-started})
            for i,id_ in enumerate(ids):
              for h in range(1,horizon+1):
                actual=p.loc[id_].iloc[origin+h-1]
                if pd.notna(actual): rows.append(dict(split=split,horizon=horizon,step=h,origin=str(dates[origin-1].date()),date=str(dates[origin+h-1].date()),territory_id=id_,model=name,actual=float(actual),prediction=float(pred[i,h-1])))
            pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False)
            print(split,horizon,origin,name,round(time.perf_counter()-started,2),flush=True)
    df=pd.DataFrame(rows); metrics=[]
    for (split,h,name),g in df.groupby(['split','horizon','model']):
        metrics.append(dict(split=split,horizon=h,model=name,n=len(g),MAE=mean_absolute_error(g.actual,g.prediction),R2=r2_score(g.actual,g.prediction)))
    m=pd.DataFrame(metrics);m.to_csv(out/'metrics.csv',index=False);pd.DataFrame(timings).to_csv(out/'timings.csv',index=False)
    selections={};boot=[]
    for h in c['horizons']:
        val=m[(m.split=='validation')&(m.horizon==h)]
        selected=val.sort_values('MAE').iloc[0].model;selections[str(h)]=selected
        test=df[(df.split=='test')&(df.horizon==h)]
        k=['territory_id','origin','date','step'];a=test[test.model==selected];b=test[test.model=='prophet']
        z=a.merge(b,on=k,suffixes=('_a','_b'))
        z['improvement']=abs(z.actual_b-z.prediction_b)-abs(z.actual_a-z.prediction_a)
        # Cluster bootstrap municipalities, keeping all time observations together.
        grouped=z.groupby('territory_id').improvement.agg(['sum','count']).to_numpy();rng=np.random.default_rng(c['seed'])
        samples=[grouped[rng.integers(0,len(grouped),len(grouped))].sum(axis=0) for _ in range(c['bootstrap_repeats'])]
        means=np.array([a/b for a,b in samples])
        boot.append({'horizon':h,'selected':selected,'MAE_gain_rub':float(z.improvement.mean()),'ci_low':float(np.quantile(means,.025)),'ci_high':float(np.quantile(means,.975))})
    (out/'selection.json').write_text(json.dumps(selections,indent=2),encoding='utf-8');pd.DataFrame(boot).to_csv(out/'bootstrap.csv',index=False)
    # Historical-origin forecast for Jan-Mar 2025. Explicitly not a live 2026 forecast.
    for h in c['horizons']:
        name=selections[str(h)];ys=p.loc[ids].T.ffill().T.to_numpy(float)
        if name in ['ridge','boosting','boosting_news']:
            pred=pooled_predict(p,24,ids,h,'ridge' if name=='ridge' else 'boosting',c,news if name=='boosting_news' else None)
        elif name=='chronos':pred=np.maximum(0,chronos.predict(torch.tensor(ys,dtype=torch.float32),prediction_length=h).cpu().numpy()[:,4,:])
        else:pred=np.array([np.repeat(y[-1],h) if name=='last' else y[-12:-12+h] if name=='seasonal' else seasonal_scaled(y,h) if name=='seasonal_scaled' else prophet_predict(y,dates,h,c) for y in ys])
        pd.DataFrame([{'territory_id':id_,'date':str((dates[-1]+pd.DateOffset(months=j+1)).date()),'prediction':float(pred[i,j]),'model':name} for i,id_ in enumerate(ids) for j in range(h)]).to_csv(out/f'forecast_h{h}.csv',index=False)
    print(m.to_string(index=False))

if __name__=='__main__':main()
