"""Causal detectors. Each call consumes observations in temporal order."""
import numpy as np

def detect(z,method,threshold,cooldown=3):
    plus=minus=ewma=0.;last=-1000;alarms=[];scores=[]
    for t,v in enumerate(z):
        if not np.isfinite(v):scores.append(float('nan'));continue
        if method=='cusum':
            plus=max(0.,plus+v-.5);minus=max(0.,minus-v-.5);score=max(plus,minus)
        elif method=='ewma':
            ewma=.3*v+.7*ewma;score=abs(ewma)/np.sqrt(.3/1.7)
        else:score=abs(v)
        scores.append(score)
        if score>threshold and t-last>cooldown:
            alarms.append(t);last=t;plus=minus=ewma=0.
    return alarms,scores

def synthetic_benchmark(seed,n=300):
    """Independent Gaussian stress test, half controls; not real shock labels."""
    rng=np.random.default_rng(seed);cases=[]
    for i in range(n):
        z=rng.normal(size=120);cp=None;kind='control'
        if i%2:
            cp=80;kind=['step_small','step_large','ramp'][i%3];sign=rng.choice([-1,1])
            z[cp:]+=sign*(1.5 if kind=='step_small' else 3 if kind=='step_large' else np.minimum(np.arange(40)*.3,3))
        cases.append((z,cp,kind))
    return cases

def evaluate_detector(cases,method,threshold):
    tp=fp=events=0;delays=[]
    for z,cp,kind in cases:
        alarms,_=detect(z,method,threshold,cooldown=12)
        if cp is not None:events+=1
        matched=False
        for t in alarms:
            if cp is not None and cp<=t<=cp+12 and not matched:
                matched=True;tp+=1;delays.append(t-cp)
            else:fp+=1
    prec=tp/max(tp+fp,1);rec=tp/max(events,1)
    return {'precision':prec,'recall':rec,'F1':2*prec*rec/max(prec+rec,1e-12),'delay_steps':float(np.mean(delays)) if delays else None,'false_alarms_per_100_steps':100*fp/(len(cases)*120),'tp':tp,'fp':fp,'events':events}
