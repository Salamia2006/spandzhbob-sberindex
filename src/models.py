"""Local baselines and pooled direct multi-horizon forecasts."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import news_features

def seasonal_scaled(y, h):
    y = np.asarray(y, dtype=float)
    growth = np.median(y[-3:] / np.maximum(y[-15:-12], 1.)) if len(y) >= 15 else y[-1] / max(y[-13], 1.) if len(y)>12 else 1.
    return np.maximum(0, y[len(y)-12:len(y)-12+h] * np.clip(growth, .5, 2.))

def feature(y, target_month, step, news=None, cutoff=None):
    y=np.asarray(y, dtype=float)
    base=max(y[-1], 1.)
    vals=[y[-k]/base for k in [1,2,3,6,12]]
    vals += [np.mean(y[-3:])/base, np.std(y[-6:])/base, np.log1p(base), step,
             np.sin(2*np.pi*target_month/12), np.cos(2*np.pi*target_month/12),
             y[len(y)-12+step-1]/base]
    if news is not None:
        vals += news[cutoff] if isinstance(news,dict) else news_features(news, cutoff)
    return vals

def pooled_predict(panel, origin, ids, horizon, method, config, news=None):
    X, target = [], []
    dates = pd.to_datetime(panel.columns)
    if news is not None:
        news={d+pd.offsets.MonthEnd(0):news_features(news,d+pd.offsets.MonthEnd(0)) for d in dates[:origin]}
    for row in panel.to_numpy(float):
        for t in range(12, origin):
            for h in range(1, horizon+1):
                if t+h>origin or not np.isfinite(row[:t+h]).all():
                    continue
                X.append(feature(row[:t], dates[t+h-1].month,h,news,dates[t-1]+pd.offsets.MonthEnd(0)))
                target.append(row[t+h-1]/max(row[t-1],1.))
    if method=='ridge':
        model=make_pipeline(StandardScaler(), Ridge(alpha=config['ridge_alpha']))
    else:
        model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=config['gb_iterations'],max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=10,random_state=config['seed'])
    model.fit(X,target)
    predictions=[]
    for id_ in ids:
        row=panel.loc[id_].iloc[:origin].to_numpy(float)
        # Missing past observations are forward filled causally at inference.
        row=pd.Series(row).ffill().to_numpy()
        f=[feature(row,(dates[origin-1]+pd.DateOffset(months=h)).month,h,news,dates[origin-1]+pd.offsets.MonthEnd(0)) for h in range(1,horizon+1)]
        predictions.append(np.maximum(0,model.predict(f)*row[-1]))
    return np.array(predictions)

def prophet_predict(y, dates, h, config):
    from prophet import Prophet
    model=Prophet(yearly_seasonality=3,weekly_seasonality=False,daily_seasonality=False,
                  changepoint_prior_scale=config['prophet_changepoint_prior_scale'],uncertainty_samples=0)
    model.fit(pd.DataFrame({'ds':dates,'y':y}))
    future=pd.date_range(pd.Timestamp(dates[-1])+pd.offsets.MonthBegin(1),periods=h,freq='MS')
    return np.maximum(0,model.predict(pd.DataFrame({'ds':future})).yhat.to_numpy())
