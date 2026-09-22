import unittest,json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import news_features,load_panel
from src.models import pooled_predict
from src.detection import detect

class IntegrityTests(unittest.TestCase):
 def test_future_news_hidden(self):
  news=pd.DataFrame({'published_at':['2024-01-10','2024-02-01'],'rate':[10,99],'delta':[1,89]})
  self.assertEqual(news_features(news,'2024-01-31'),[10,1,1])
  self.assertEqual(news_features(news,'2024-01-01'),[0,0,0])
 def test_causal_detector(self):
  rng=np.random.default_rng(42);z=rng.normal(size=100)
  for method in ['cusum','ewma','threshold']:
   a,s=detect(z,method,4);b,t=detect(z[:60],method,4)
   self.assertEqual([i for i in a if i<60],b);np.testing.assert_allclose(s[:60],t)
 def test_no_future_training_leak(self):
  c=json.loads(Path('config.json').read_text(encoding='utf-8'));p,ids,_=load_panel('data/consumption.parquet',c)
  p=p.iloc[:40];ids=list(p.index[:3]);a=pooled_predict(p,15,ids,3,'ridge',c)
  changed=p.copy();changed.iloc[:,15:]=999999
  b=pooled_predict(changed,15,ids,3,'ridge',c);np.testing.assert_allclose(a,b)
 def test_selection_and_shared_rows(self):
  if not Path('results/metrics.csv').exists():self.skipTest('Run benchmark first')
  d=pd.read_csv('results/predictions.csv');m=pd.read_csv('results/metrics.csv');s=json.loads(Path('results/selection.json').read_text())
  self.assertFalse(d.duplicated(['split','horizon','step','origin','date','territory_id','model']).any())
  self.assertTrue(np.isfinite(d.prediction).all())
  for (split,h),g in d.groupby(['split','horizon']):self.assertEqual(g.groupby('model').size().nunique(),1)
  for h,name in s.items():self.assertEqual(name,m[(m.split=='validation')&(m.horizon==int(h))].sort_values('MAE').iloc[0].model)
  for _,r in m.iterrows():
   g=d[(d.split==r.split)&(d.horizon==r.horizon)&(d.model==r.model)]
   self.assertAlmostEqual(float(abs(g.actual-g.prediction).mean()),r.MAE,places=7)

if __name__=='__main__':unittest.main()
