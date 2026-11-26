"""B2: gradient-boosted trees on the same feature vector the deep models get (README §3).

HistGradientBoostingRegressor with absolute-error loss, predicting y (actual - typical time); the prediction is
t + base + y_hat. Inputs are FEATURES from rows.py plus route (categorical) and direction. mta_pred is never an input.
Missing values (no train ahead, short history) are handled natively by the trees.

Trained on train.parquet (optionally plus history.parquet). The number of trees is chosen on the validation week:
the validation MAE is tracked after every tree and the model is cut at the best one.

Usage:  python baselines.py                → data/models/B2.joblib, data/preds/val/B2.parquet
        python baselines.py --history      also train on history.parquet rows
        python baselines.py --predict test  predictions for another split with the saved model
then:   python score.py val
"""
import argparse, os, time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from rows import FEATURES

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
KEY = ['trip', 'p', 'stop_id_b']
INPUTS = FEATURES + ['route', 'direction']
MODEL = os.path.join(DATA, 'models', 'B2.joblib')


def load(split, routes, sample=None, seed=0):
    df = pd.read_parquet(os.path.join(DATA, 'rows', f'{split}.parquet'), columns=KEY + INPUTS + ['t', 'y'])
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=seed).reset_index(drop=True)
    X = df[INPUTS].copy()
    X['route'] = df.route.map({r: i for i, r in enumerate(routes)}).astype('float32')   # unseen line → missing
    return df, X.astype('float32')


def write_preds(split, df, y_hat):
    out = os.path.join(DATA, 'preds', split)
    os.makedirs(out, exist_ok=True)
    pred = df.t.to_numpy('float64') + df.base.to_numpy('float64') + y_hat
    df[KEY].assign(pred=pred).to_parquet(os.path.join(out, 'B2.parquet'), index=False)
    print(f'wrote {os.path.join(out, "B2.parquet")} ({len(df):,} rows)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--history', action='store_true', help='add history rows to the training set')
    ap.add_argument('--trees', type=int, default=500)
    ap.add_argument('--predict', help='only predict this split with the saved model')
    args = ap.parse_args()
    t0 = time.time()

    if args.predict:
        saved = joblib.load(MODEL)
        df, X = load(args.predict, saved['routes'])
        write_preds(args.predict, df, saved['model'].predict(X))
        return

    routes = sorted(pd.read_parquet(os.path.join(DATA, 'rows', 'train.parquet'), columns=['route']).route.unique())
    tr, Xtr = load('train', routes)
    ytr = tr.y.to_numpy('float32')
    if args.history:
        hi, Xhi = load('history', routes, sample=4_000_000)
        Xtr, ytr = pd.concat([Xtr, Xhi], ignore_index=True), np.r_[ytr, hi.y.to_numpy('float32')]
        del hi, Xhi
    del tr
    print(f'training on {len(Xtr):,} rows, {len(INPUTS)} inputs ({time.time() - t0:.0f} s)')

    cat = np.array([c == 'route' for c in INPUTS])
    model = HistGradientBoostingRegressor(loss='absolute_error', max_iter=args.trees, learning_rate=0.1,
                                          max_leaf_nodes=63, min_samples_leaf=200, categorical_features=cat,
                                          early_stopping=False, random_state=0, verbose=1)
    model.fit(Xtr, ytr)
    del Xtr, ytr
    print(f'fitted {model.n_iter_} trees ({time.time() - t0:.0f} s)')

    va, Xva = load('val', routes)
    yva = va.y.to_numpy('float64')
    curve, best, best_pred = [], None, None
    for i, y_hat in enumerate(model.staged_predict(Xva), 1):
        mae = np.abs(y_hat - yva).mean()
        curve.append(mae)
        if i % 50 == 0:
            print(f'  {i} trees: validation MAE {mae:.1f} s', flush=True)
        if best is None or mae < curve[best - 1]:
            best, best_pred = i, y_hat
    print('validation MAE (s) by number of trees: '
          + ', '.join(f'{i}: {curve[i - 1]:.1f}' for i in sorted({1, 50, 100, 200, 300, 400, len(curve), best})
                      if i <= len(curve)))
    print(f'best: {best} trees, validation MAE {curve[best - 1]:.1f} s (B1 typical times: {np.abs(yva).mean():.1f} s)')

    # keep only the first `best` trees, so the saved model predicts exactly what was validated
    model._predictors = model._predictors[:best]                 # n_iter_ follows the list length
    assert np.allclose(model.predict(Xva[:1000]), best_pred[:1000], atol=1e-3)
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    joblib.dump({'model': model, 'routes': routes, 'inputs': INPUTS, 'history': args.history}, MODEL)
    print(f'wrote {MODEL}')
    write_preds('val', va, best_pred)
    print(f'done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
