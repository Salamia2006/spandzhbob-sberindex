"""Проверка первичного набора, выбор панели только по обучающему периоду."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

def load_panel(path, config):
    d = pd.read_parquet(path)
    d.columns = d.columns.str.strip()
    d['category'] = d.category.str.strip()
    if 'value' not in d and 'consumption' in d:
        d = d.rename(columns={'consumption': 'value'})
    assert not d.duplicated(['date', 'territory_id', 'category']).any(), 'Duplicate keys'
    assert d.value.notna().all() and (d.value >= 0).all(), 'Invalid values'
    p = d[d.category == config['category']].pivot(index='territory_id', columns='date', values='value').sort_index(axis=1)
    # Eligibility depends on Jan 2023-Mar 2024 only; never on test outcomes.
    eligible = p.iloc[:, :15].notna().all(axis=1) & (p.iloc[:, :15] > 0).all(axis=1)
    p = p[eligible]
    rng = np.random.default_rng(config['seed'])
    ids = sorted(rng.choice(p.index, min(config['evaluation_municipalities'], len(p)), replace=False).tolist())
    quality = {'raw_rows': len(d), 'raw_territories': int(d.territory_id.nunique()),
               'panel_territories': len(p), 'months': list(p.columns), 'evaluation_ids': ids,
               'missing_panel_values': int(p.isna().sum().sum()),
               'source_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    return p, ids, quality

def news_features(news, cutoff):
    """Only already published announcements are visible. No backward fill."""
    known = news[pd.to_datetime(news.published_at) <= pd.Timestamp(cutoff)]
    if known.empty:
        return [0., 0., 0.]
    last = known.sort_values('published_at').iloc[-1]
    recent = known[pd.to_datetime(known.published_at) > pd.Timestamp(cutoff) - pd.Timedelta(days=31)]
    return [float(last.rate), float(recent.delta.sum()), float(len(recent))]
