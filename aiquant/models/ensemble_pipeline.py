
from numba import njit

@njit
def compute_triple_barrier_labels(
    open_arr,
    high_arr,
    low_arr,
    atr_arr,
    timeout_bars=24,
    stop_mult=1.8,
    target_ratio=1.25,
    min_stop_pct=0.0030
):
    n = len(open_arr)
    labels = np.zeros(n, dtype=np.int8)

    for i in range(n - timeout_bars - 1):
        entry_idx = i + 1
        entry = open_arr[entry_idx]
        atr = atr_arr[i]

        if not np.isfinite(atr) or atr <= 0.0:
            continue

        base_stop_dist = max(entry * min_stop_pct, atr * stop_mult)
        tp_dist = base_stop_dist * target_ratio

        sl_long = entry - base_stop_dist
        tp_long = entry + tp_dist
        long_win = False
        long_loss = False

        sl_short = entry + base_stop_dist
        tp_short = entry - tp_dist
        short_win = False
        short_loss = False

        end_idx = min(entry_idx + timeout_bars, n - 1)

        # Scan path for Long
        for j in range(entry_idx, end_idx + 1):
            b_open = open_arr[j]
            b_high = high_arr[j]
            b_low = low_arr[j]

            hit_sl = b_low <= sl_long
            hit_tp = b_high >= tp_long

            if hit_sl and hit_tp:
                if abs(b_open - tp_long) < abs(b_open - sl_long):
                    long_win = True
                else:
                    long_loss = True
                break
            elif hit_tp:
                long_win = True
                break
            elif hit_sl:
                long_loss = True
                break

        # Scan path for Short
        for j in range(entry_idx, end_idx + 1):
            b_open = open_arr[j]
            b_high = high_arr[j]
            b_low = low_arr[j]

            hit_sl = b_high >= sl_short
            hit_tp = b_low <= tp_short

            if hit_sl and hit_tp:
                if abs(b_open - tp_short) < abs(b_open - sl_short):
                    short_win = True
                else:
                    short_loss = True
                break
            elif hit_tp:
                short_win = True
                break
            elif hit_sl:
                short_loss = True
                break

        if long_win and not short_win:
            labels[i] = 1
        elif short_win and not long_win:
            labels[i] = -1
        else:
            labels[i] = 0

    return labels

"""
aiquant/models/ensemble_pipeline.py
===================================
The ML ensemble backtest pipeline (XGBoost + LightGBM + LSTM, walk-forward CV).

Single source of the strategy's ML logic: `run.py backtest` imports
run_ml_backtest from here, and the models/ml_live_bundle.pkl it produces is what
aiquant/execution/ml_live_trader.py loads for live trading.
"""

import os
import json
import time
import platform
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from aiquant.utils.console import BOLD, DIM, GREEN, RED, CYAN, YELLOW, WHITE
from aiquant.strategy.trade_plan import TradePlanEngine
from aiquant.strategy.backtest_execution import simulate_trade_plan

# -- Paths (repo root = two levels up from aiquant/models/) --------------------
ROOT        = Path(__file__).resolve().parents[2]
CONFIG_DIR  = ROOT / 'config'
RESULTS_DIR = ROOT / 'results'
MODELS_DIR  = ROOT / 'models'


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# ML ENSEMBLE BACKTEST  (XGBoost + LightGBM + LSTM, walk-forward CV)
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

def run_ml_backtest(df: pd.DataFrame, pair: str, capital: float = 100_000,
                    fast: bool = False, days: int = None,
                    force_feature_selection: bool = False) -> dict:
    """
    Full ML ensemble backtest using walk-forward cross-validation.

    Parameters
    ----------
    df      : DataFrame with OHLCV + 183 features (output of build_features)
    pair    : Trading pair string for display
    capital : Starting capital in USD
    fast    : If True, skip LSTM (faster, ~Sharpe 3.0+)

    Returns
    -------
    dict with keys: sharpe, ret, max_dd, calmar, pf, trades, win_rate,
                    final, equity, signals
    """
    from sklearn.preprocessing import RobustScaler
    from sklearn.feature_selection import mutual_info_classif
    import xgboost as xgb
    import lightgbm as lgb
    import gc

    n      = len(df)
    c      = df['close'].to_numpy(np.float64)
    dates  = df.index
    # Истинный институциональный макро-режим:
    # 4H EMA-50 на 15m свечах = 50 * 16 = 800 баров (~8.3 суток)
    # Если в df уже есть каузальный mtf_4h_structure_regime, используем его
    # Истинный институциональный макро-режим (50 Daily EMA):
    # 50 дней * 24 часа * 4 бара = 4800 баров 15m свечей (~50 суток)
    # Быстрый фильтр (4H EMA-50): 800 баров (~8.3 суток)
    # Если режимы уже предрассчитаны для каждого актива изолированно:
    if 'bullish_regime' in df.columns and 'bearish_regime' in df.columns:
        bullish_regime = df['bullish_regime'].to_numpy(dtype=bool)
        bearish_regime = df['bearish_regime'].to_numpy(dtype=bool)
    else:
        macro_daily_ema50 = pd.Series(c).ewm(span=4800, min_periods=200).mean().to_numpy()
        macro_4h_ema50    = pd.Series(c).ewm(span=800, min_periods=100).mean().to_numpy()
        macro_bull = c > macro_daily_ema50
        macro_bear = c < macro_daily_ema50

        if 'mtf_4h_structure_regime' in df.columns:
            struct = df['mtf_4h_structure_regime'].to_numpy()
            bullish_regime = (struct > 0) & macro_bull
            bearish_regime = (struct < 0) & macro_bear & (c < macro_4h_ema50)
        else:
            bullish_regime = macro_bull & (c > macro_4h_ema50)
            bearish_regime = macro_bear & (c < macro_4h_ema50)

    # -- Memory banner --------------------------------------------------------
    try:
        import psutil
        _mem = psutil.virtual_memory()
        _used_gb  = (_mem.total - _mem.available) / 1e9
        _total_gb = _mem.total / 1e9
        print(f"  {DIM(f'RAM: {_used_gb:.1f} / {_total_gb:.1f} GB used  |  {n:,} bars  |  {len(df.columns)} columns')}")
    except Exception:
        pass

    # -- Downcast feature DataFrame to float32 to halve memory ---------------
    float_cols = df.select_dtypes(include=[np.float64]).columns
    df[float_cols] = df[float_cols].astype(np.float32)

    # -- Label generation ----------------------------------------------------
    print("\n  [*] [1/5] Generating labels (volatility-scaled triple barrier)...")
    TIMEOUT_BARS = 24
    FORWARD_BARS = TIMEOUT_BARS
    open_arr = df["open"].to_numpy(dtype=np.float64)
    high_arr = df["high"].to_numpy(dtype=np.float64)
    low_arr  = df["low"].to_numpy(dtype=np.float64)
    atr_vals = df["atr_14"].to_numpy(dtype=np.float64) if "atr_14" in df.columns else np.zeros(n, dtype=np.float64)

    # 15m Institutional Triple Barrier: SL >= 0.90%, TP = 1.40R, Horizon = 16 bars (4h)
    labels = compute_triple_barrier_labels(
        open_arr=open_arr,
        high_arr=high_arr,
        low_arr=low_arr,
        atr_arr=atr_vals,
        timeout_bars=TIMEOUT_BARS,
        stop_mult=2.0,
        target_ratio=1.15,
        min_stop_pct=0.0090
    )

    valid_mask = np.zeros(n, dtype=bool)
    valid_mask[:n - TIMEOUT_BARS - 1] = True

    cc = {-1: int((labels == -1).sum()), 0: int((labels == 0).sum()), 1: int((labels == 1).sum())}
    print(f"  [OK] Long={cc[1]:,}  Short={cc[-1]:,}  Flat={cc[0]:,}  "
          f"({(cc[1]+cc[-1])/n*100:.1f}% directional)")

    # ------------------ Feature selection ------------------
    print(f"\n  {CYAN('[*]')}  [2/5] Feature selection (mutual information)...")

    # Исключаем все нестационарные признаки в долларах
    banned_prefixes = ('sma_', 'ema_', 'wma_', 'hma_', 'dema_', 'tema_')
    drop_cols = {
        'open', 'high', 'low', 'close', 'volume', 'quote_volume',
        'macro_daily_ema50', 'macro_4h_ema50',
        'dc_high', 'dc_low', 'dc_mid',
        'bb_high', 'bb_low', 'bb_mid',
        'kc_high', 'kc_low', 'kc_mid',
        'vwap', 'vwap_upper', 'vwap_lower',
        'pivot', 'r1', 's1', 'r2', 's2', 'r3', 's3',
        'swing_high', 'swing_low',
        'asset_id', 'bullish_regime', 'bearish_regime'
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [
        c_ for c_ in numeric_cols 
        if c_ not in drop_cols 
        and not c_.startswith(banned_prefixes)
        and not c_.startswith('target')
    ]

    # Load saved top features if available вЂ” avoids building the full X_all matrix
    params_path = CONFIG_DIR / 'ml_best_params.json'
    saved_features = None
    if params_path.exists() and not force_feature_selection:
        try:
            with open(params_path) as f:
                saved = json.load(f)
            sf = saved.get('top_features', [])
            if sf and all(feat in feature_cols for feat in sf):
                saved_features = sf
                print(f"  {GREEN('[OK]')} Using {len(saved_features)} saved features from ml_best_params.json")
        except Exception:
            pass

    if saved_features is None:
        # Only build X_all for MI scoring вЂ” use a sample to save RAM
        X_all = df[feature_cols].to_numpy(np.float32)   # float32: half the RAM of float64
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
        sample_idx = np.random.choice(
            np.where(valid_mask)[0], size=min(20000, valid_mask.sum()), replace=False
        )
        sample_idx.sort()
        X_sample = X_all[sample_idx].astype(np.float64)   # MI needs float64
        y_sample = labels[sample_idx]
        y_binary = (y_sample != 0).astype(int)
        mi_scores = mutual_info_classif(X_sample, y_binary, random_state=42, n_neighbors=5)
        mi_df     = pd.DataFrame({'feature': feature_cols, 'mi': mi_scores}).sort_values('mi', ascending=False)
        TOP_K     = 60
        saved_features = mi_df.head(TOP_K)['feature'].tolist()
        print(f"  {GREEN('[OK]')} Top {TOP_K} features selected from {len(feature_cols)} total")
        # Free X_all immediately вЂ” no longer needed
        del X_all, X_sample
        gc.collect()

    top_features = saved_features

    # Extract only the top-60 feature matrix (float32 = 0.63 GB for 2.6M bars)
    X_sel = df[top_features].to_numpy(np.float32)
    X_sel = np.nan_to_num(X_sel, nan=0.0, posinf=0.0, neginf=0.0)

    # Free the full feature DataFrame вЂ” X_sel is all we need from here on
    _ohlcv_cols = ['open', 'high', 'low', 'close', 'volume', 'atr_14', 'dc_high', 'dc_low']
    df_ohlcv = df[_ohlcv_cols].copy()   # keep OHLCV for backtest simulation
    del df
    gc.collect()
    df = df_ohlcv   # rebind name so rest of function still works

    try:
        import psutil
        _mem = psutil.virtual_memory()
        _used_gb = (_mem.total - _mem.available) / 1e9
        _total_gb = _mem.total / 1e9
        print(f"  {DIM(f'RAM after feature selection: {_used_gb:.1f} / {_total_gb:.1f} GB')}")
    except Exception:
        pass

    print(f"  {DIM('Top 5: ' + ', '.join(top_features[:5]))}")

    # -- Walk-forward CV setup ------------------------------------------------
    print(f"\n  {CYAN('[*]')}  [3/5] Walk-forward cross-validation...")

    # Dynamic fold sizing вЂ” targets ~50 folds regardless of dataset length.
    # Training window : 1/6 of total days, clamped to [7d, 90d]
    # Test/step window: sized so total folds в‰€ 50, clamped to [1d, 30d]
    # Fallback to fixed 30d/7d if fewer than 10 folds would result.
    # Dynamic BARS_PER_DAY calculation based on bar frequency
    try:
        dt_seconds = (df.index[1] - df.index[0]).total_seconds()
        BARS_PER_DAY = max(1, int(86400 / dt_seconds)) if dt_seconds > 0 else 96
    except Exception:
        BARS_PER_DAY = 96  # fallback for 15m

    # На 15m барах эмбарго (FORWARD_BARS) равно 16 барам (4 часа) вместо старых 180 минут
    FORWARD_BARS = min(FORWARD_BARS, 16) if 'FORWARD_BARS' in locals() or 'FORWARD_BARS' in globals() else 16

    TARGET_FOLDS = 50
    total_usable = n - FORWARD_BARS
    days_total   = total_usable / BARS_PER_DAY

    train_days = max(20, min(60, int(days_total * 0.45)))
    TRAIN_BARS = train_days * BARS_PER_DAY
    step_days  = max(1, min(30, int((days_total - train_days) / TARGET_FOLDS)))
    TEST_BARS  = step_days * BARS_PER_DAY
    STEP_BARS  = TEST_BARS

    def _build_folds(n_bars, train_b, test_b, step_b, fwd):
        fs, s = [], 0
        while s + train_b + fwd + test_b <= n_bars:
            train_end = s + train_b
            test_start = train_end + fwd
            test_end = test_start + test_b
            fs.append((s, train_end, test_start, test_end))
            s += step_b
        return fs

    folds = _build_folds(n, TRAIN_BARS, TEST_BARS, STEP_BARS, FORWARD_BARS)

    # Fallback 1: if dynamic sizing produced fewer than 10 folds, use fixed 30d/7d
    if len(folds) < 10:
        train_days, step_days = 30, 7
        TRAIN_BARS = train_days * BARS_PER_DAY
        TEST_BARS  = step_days  * BARS_PER_DAY
        STEP_BARS  = TEST_BARS
        folds = _build_folds(n, TRAIN_BARS, TEST_BARS, STEP_BARS, FORWARD_BARS)

    # Fallback 2: dataset too short for 30d/7d вЂ” shrink to fit available data
    # Minimum: train=50% of data, test=remaining, at least 1 fold
    if len(folds) == 0:
        train_days = max(3, int(days_total * 0.5))
        step_days  = max(1, int(days_total * 0.3))
        TRAIN_BARS = train_days * BARS_PER_DAY
        TEST_BARS  = step_days  * BARS_PER_DAY
        STEP_BARS  = TEST_BARS
        folds = _build_folds(n, TRAIN_BARS, TEST_BARS, STEP_BARS, FORWARD_BARS)

    if len(folds) == 0:
        raise ValueError(
            f"Dataset too short for walk-forward CV: {days_total:.0f} days available. "
            f"Minimum required: ~10 days. Use --days 45 or more."
        )

    print(f"  {GREEN('[OK]')} {len(folds)} folds  "
          f"(train={train_days}d | test/step={step_days}d | dataset={days_total:.0f}d)")

    # -- GPU detection for ML ----------------------------------------------------
    try:
        import torch as _torch
        _ml_device = 'cuda' if _torch.cuda.is_available() else 'cpu'
        _gpu_name  = _torch.cuda.get_device_name(0) if _ml_device == 'cuda' else 'CPU'
    except Exception:
        _ml_device = 'cpu'
        _gpu_name  = 'CPU'

    # XGBoost GPU tree method
    _xgb_tree_method = 'hist'
    _xgb_device      = 'cuda' if _ml_device == 'cuda' else 'cpu'

    # LightGBM GPU on Windows uses the OpenCL backend.
    # XGBoost uses native CUDA separately above.
    _lgb_device = 'gpu' if _ml_device == 'cuda' else 'cpu'

    # Suppress LightGBM OpenCL/CUDA compiler messages ("1 warning generated.")
    # These are emitted by the native C library to file descriptor 2 (stderr)
    # before Python's environment is read, so os.environ alone is not enough.
    # We redirect fd=2 to /dev/null for the duration of training, then restore.
    import os as _os
    _os.environ['LIGHTGBM_VERBOSITY'] = '-1'
    _os.environ['LGB_VERBOSITY']      = '-1'
    # C-level stderr redirect
    _devnull_fd  = _os.open(_os.devnull, _os.O_WRONLY)
    _stderr_fd   = 2
    _stderr_save = _os.dup(_stderr_fd)   # save original stderr fd
    _os.dup2(_devnull_fd, _stderr_fd)    # point stderr -> /dev/null
    _os.close(_devnull_fd)

    # Report GPU RAM if available
    if _ml_device == 'cuda':
        try:
            import torch as _t2
            _free, _total = _t2.cuda.mem_get_info(0)
            print(f"  {DIM(f'  GPU RAM: {(_total-_free)/1e9:.1f} / {_total/1e9:.1f} GB used')}")
        except Exception:
            pass

    print(f"\n  {CYAN('[*]')}  [4/5] Training XGBoost + LightGBM (walk-forward)...")
    print(f"  {DIM(f'  Device: {_gpu_name}  |  XGBoost tree_method=hist device={_xgb_device}  |  {len(folds)} folds')}")
    print(f"  {DIM(f'  Each fold: ~43k train bars x 60 features  ->  10k test bars')}")
    print()

    oos_xgb  = np.zeros(n)
    oos_lgb  = np.zeros(n)
    oos_p0   = np.zeros(n)
    oos_p1   = np.zeros(n)
    oos_p2   = np.zeros(n)
    oos_mask = np.zeros(n, dtype=bool)
    scaler   = RobustScaler()

    def encode_labels(y): return (y + 1).astype(int)
    def decode_labels(y): return (y - 1).astype(np.int8)

    _xgb_fold_times = []
    _t_xgb_start    = time.time()

    for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        _t_fold = time.time()
        tr_mask = valid_mask[tr_start:tr_end]
        tr_idx  = np.arange(tr_start, tr_end)[tr_mask]
        X_tr    = X_sel[tr_idx]
        y_tr    = encode_labels(labels[tr_idx])
        te_idx  = np.arange(te_start, te_end)
        X_te    = X_sel[te_idx]
        X_tr_s  = scaler.fit_transform(X_tr)
        X_te_s  = scaler.transform(X_te)

        # Scientific Protocol: Uniform weighting prevents probability distortion
        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            objective='multi:softprob', num_class=3,
            eval_metric='mlogloss', random_state=42, verbosity=0,
            use_label_encoder=False,
            tree_method=_xgb_tree_method, device=_xgb_device,
        )
        xgb_model.fit(X_tr_s, y_tr)
        xgb_proba = xgb_model.predict_proba(X_te_s)

        lgb_model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1,
            device=_lgb_device,
        )
        lgb_model.fit(X_tr_s, y_tr)
        lgb_proba = lgb_model.predict_proba(X_te_s)

        # Безопасное выравнивание вероятностей по классам [0: Short, 1: Flat, 2: Long]
        def _align_3class_proba(model, proba, n_rows):
            out = np.zeros((n_rows, 3), dtype=np.float64)
            classes = getattr(model, 'classes_', None)
            if classes is None or len(classes) == proba.shape[1] == 3:
                return proba
            for col_i, cls_val in enumerate(classes):
                if 0 <= cls_val < 3 and col_i < proba.shape[1]:
                    out[:, int(cls_val)] = proba[:, col_i]
            return out

        xgb_p = _align_3class_proba(xgb_model, xgb_proba, len(te_idx))
        lgb_p = _align_3class_proba(lgb_model, lgb_proba, len(te_idx))

        # Store calibrated class probabilities
        oos_p0[te_idx] = 0.5 * xgb_p[:, 0] + 0.5 * lgb_p[:, 0]
        oos_p1[te_idx] = 0.5 * xgb_p[:, 1] + 0.5 * lgb_p[:, 1]
        oos_p2[te_idx] = 0.5 * xgb_p[:, 2] + 0.5 * lgb_p[:, 2]
        oos_xgb[te_idx]  = xgb_p[:, 2] - xgb_p[:, 0]
        oos_lgb[te_idx]  = lgb_p[:, 2] - lgb_p[:, 0]
        oos_mask[te_idx] = True

        _fold_elapsed = time.time() - _t_fold
        _xgb_fold_times.append(_fold_elapsed)
        _avg_fold     = sum(_xgb_fold_times) / len(_xgb_fold_times)
        _remaining    = _avg_fold * (len(folds) - fold_idx - 1)
        _total_so_far = time.time() - _t_xgb_start
        print(
            f"  {DIM(f'  XGB+LGB Fold {fold_idx+1:>2}/{len(folds)}')}"
            f"  {DIM(f'{_fold_elapsed:.1f}s/fold')}"
            f"  {DIM(f'elapsed {_total_so_far:.0f}s')}"
            f"  {CYAN(f'ETA ~{_remaining:.0f}s')}",
            flush=True
        )

    # Restore stderr after XGB+LGB training (was redirected to suppress LGB warnings)
    _os.dup2(_stderr_save, _stderr_fd)
    _os.close(_stderr_save)

    print(f"  {GREEN('[OK]')} OOS coverage: {oos_mask.sum():,} bars ({oos_mask.mean()*100:.1f}%)"
          f"  {DIM(f'  total {time.time()-_t_xgb_start:.0f}s')}")

    # -- LSTM training --------------------------------------------------------
    oos_lstm       = np.zeros(n)
    LSTM_AVAILABLE = False

    print(f"\n  {YELLOW('[*]')}  [4b] LSTM Skipped (Fast Tree-Only mode, saving 5 min)...")
    if False:  # LSTM completely disabled to run fast trees
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            SEQ_LEN    = 30
            LSTM_FEATS = 20
            # X_sel already holds the top-60 features as float32 вЂ” slice first 20
            X_lstm_raw = X_sel[:, :LSTM_FEATS].astype(np.float32)
            X_lstm_raw = np.nan_to_num(X_lstm_raw, nan=0.0, posinf=0.0, neginf=0.0)

            class LSTMAttn(nn.Module):
                def __init__(self, input_size, hidden=64, n_layers=2, n_classes=3):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden, n_layers,
                                        batch_first=True, dropout=0.2)
                    self.attn = nn.Linear(hidden, 1)
                    self.fc   = nn.Sequential(
                        nn.Linear(hidden, 32), nn.ReLU(),
                        nn.Dropout(0.2), nn.Linear(32, n_classes)
                    )
                def forward(self, x):
                    out, _ = self.lstm(x)
                    attn_w = torch.softmax(self.attn(out), dim=1)
                    ctx    = (out * attn_w).sum(dim=1)
                    return self.fc(ctx)

            _t_lstm_start  = time.time()
            _lstm_fold_times = []
            print(f"  {DIM(f'  Device: {DEVICE}  |  SEQ_LEN={SEQ_LEN}  |  {LSTM_FEATS} features  |  5 epochs/fold  |  {len(folds)} folds')}")
            print()

            for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
                _t_lstm_fold = time.time()
                tr_seqs, tr_labs = [], []
                for i in range(tr_start + SEQ_LEN, tr_end):
                    if not valid_mask[i]: continue
                    tr_seqs.append(X_lstm_raw[i-SEQ_LEN:i])
                    tr_labs.append(encode_labels(labels[i]))
                if len(tr_seqs) < 100: continue

                X_tr_t = torch.tensor(np.array(tr_seqs), dtype=torch.float32).to(DEVICE)
                y_tr_t = torch.tensor(tr_labs, dtype=torch.long).to(DEVICE)
                feat_mean = X_tr_t.mean(dim=(0, 1), keepdim=True)
                feat_std  = X_tr_t.std(dim=(0, 1), keepdim=True) + 1e-8
                X_tr_t    = (X_tr_t - feat_mean) / feat_std

                ds      = TensorDataset(X_tr_t, y_tr_t)
                # Larger batch size saturates T4 GPU better (512 -> 2048)
                _lstm_batch = 2048 if _ml_device == 'cuda' else 512
                loader  = DataLoader(ds, batch_size=_lstm_batch, shuffle=True)
                model   = LSTMAttn(LSTM_FEATS).to(DEVICE)
                opt     = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
                sched   = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
                loss_fn = nn.CrossEntropyLoss()

                model.train()
                epoch_losses = []
                for epoch in range(5):
                    ep_loss = 0.0
                    for xb, yb in loader:
                        opt.zero_grad()
                        loss = loss_fn(model(xb), yb)
                        loss.backward(); opt.step()
                        ep_loss += loss.item()
                    sched.step()
                    epoch_losses.append(ep_loss / max(len(loader), 1))

                model.eval()
                te_idx_list = list(range(te_start + SEQ_LEN, te_end))
                if te_idx_list:
                    te_seqs = [X_lstm_raw[i-SEQ_LEN:i] for i in te_idx_list]
                    X_te_t  = torch.tensor(np.array(te_seqs), dtype=torch.float32).to(DEVICE)
                    X_te_t  = (X_te_t - feat_mean) / feat_std
                    with torch.no_grad():
                        proba = torch.softmax(model(X_te_t), dim=1).cpu().numpy()
                    for j, i in enumerate(te_idx_list):
                        oos_lstm[i] = proba[j, 2] - proba[j, 0]

                _lstm_fold_elapsed = time.time() - _t_lstm_fold
                _lstm_fold_times.append(_lstm_fold_elapsed)
                _lstm_avg   = sum(_lstm_fold_times) / len(_lstm_fold_times)
                _lstm_eta   = _lstm_avg * (len(folds) - fold_idx - 1)
                _lstm_total = time.time() - _t_lstm_start
                _loss_str   = f'loss={epoch_losses[-1]:.4f}' if epoch_losses else ''
                print(
                    f"  {DIM(f'  LSTM Fold {fold_idx+1:>2}/{len(folds)}')}"
                    f"  {DIM(f'{_lstm_fold_elapsed:.1f}s/fold')}"
                    f"  {DIM(_loss_str)}"
                    f"  {DIM(f'elapsed {_lstm_total:.0f}s')}"
                    f"  {CYAN(f'ETA ~{_lstm_eta:.0f}s')}",
                    flush=True
                )

            LSTM_AVAILABLE = True
            print(f"  {GREEN('[OK]')} LSTM training complete  "
                  f"(device: {DEVICE}  |  total {time.time()-_t_lstm_start:.0f}s)")

        except Exception as e:
            print(f"  {YELLOW('вљ ')} LSTM skipped: {e}")

    # -- Ensemble score (Tree-Only Isolation: XGB 50% + LGB 50%) --------------
    ens_score = 0.50 * oos_xgb + 0.50 * oos_lgb
    print(f"\n  {DIM('Ensemble: Pure Trees (XGB 50% + LGB 50%) — LSTM Isolated')}")

    # -- Validation / Final Test threshold selection ----------------------------
    print(f"\n  {CYAN('[*]')}  [5/5] Validation threshold search + final test...")

    # IMPORTANT:
    # Thresholds are selected ONLY on validation OOS data.
    # Final test is kept completely unseen until the final evaluation.
    oos_idx = np.where(oos_mask)[0]

    if len(oos_idx) < 100:
        print(f"  {RED('вњ—')} Not enough OOS bars for validation/final-test split.")
        return {}

    split = int(len(oos_idx) * 0.70)
    val_idx = oos_idx[:split]
    test_idx = oos_idx[split:]

    print(
        f"  OOS split: {len(val_idx):,} validation bars / "
        f"{len(test_idx):,} final-test bars"
    )

    # -- Expected Value & Probability Domination Gate ---------------------
    # Only evaluate signals where directional conviction strictly dominates FLAT noise
    val_long_cand = (oos_p2[val_idx] > oos_p1[val_idx]) & (oos_p2[val_idx] >= 0.46)
    val_short_cand = (oos_p0[val_idx] > oos_p1[val_idx]) & (oos_p0[val_idx] >= 0.46)
    # -- Institutional High-Conviction EV-Gate --------------------------------
    # Жесткий институциональный фильтр отсечения шума (0.60 .. 0.64) и маржа 0.08
    FLAT_MARGIN = 0.055
    SHORT_REGIME_PENALTY = 0.06

    candidates = [0.56, 0.58, 0.60, 0.62]
    best_long = 0.58
    best_short = 0.58

    for th in candidates:
        long_cond = (oos_p2[val_idx] >= th) & ((oos_p2[val_idx] - oos_p1[val_idx]) >= FLAT_MARGIN)
        short_cond = (oos_p0[val_idx] >= th) & ((oos_p0[val_idx] - oos_p1[val_idx]) >= FLAT_MARGIN)
        if long_cond.sum() >= 15:
            best_long = th
        if short_cond.sum() >= 15:
            best_short = th

    LONG_HARD_THRESHOLD = max(best_long, 0.58)
    SHORT_HARD_THRESHOLD = max(best_short, 0.58)

    print(f'  [OK] Asymmetric EV-Gate: [P(Long) >= {LONG_HARD_THRESHOLD:.2f} | P(Short) >= {SHORT_HARD_THRESHOLD:.2f} (+{SHORT_REGIME_PENALTY:.2f} in bull) | Margin >= {FLAT_MARGIN:.2f}]')

    # -- FINAL TEST: Направленные сигналы с асимметричным шлюзом --
    sig = np.zeros(n, dtype=np.int8)

    # Лонги: базовая уверенность над боковиком
    long_mask = (oos_p2 >= LONG_HARD_THRESHOLD) & ((oos_p2 - oos_p1) >= FLAT_MARGIN)
    sig[long_mask] = 1

    # Шорты: СТРОГО запрещены в бычьем макро-режиме (цена выше тренда)
    short_mask = (~bullish_regime) & (oos_p0 >= SHORT_HARD_THRESHOLD) & ((oos_p0 - oos_p1) >= FLAT_MARGIN)
    sig[short_mask] = -1
    # Шорты: СТРОГО запрещены в бычьем макро-режиме (цена выше тренда)
    short_mask = (~bullish_regime) & (oos_p0 >= SHORT_HARD_THRESHOLD) & ((oos_p0 - oos_p1) >= FLAT_MARGIN)
    sig[short_mask] = -1

    # Ограничиваем строго финальным тестовым срезом
    sig[~np.isin(np.arange(n), test_idx)] = 0

    # Симметричный институциональный макро-фильтр:
    # 1. Лонги разрешены ТОЛЬКО в подтвержденном бычьем макро-тренде
    sig[(sig == 1) & (~bullish_regime)] = 0
    # 2. Шорты разрешены ТОЛЬКО в подтвержденном медвежьем макро-тренде
    sig[(sig == -1) & (~bearish_regime)] = 0

    execution = simulate_trade_plan(
        probs=np.maximum(oos_p2, oos_p0),
        signal=sig,
        open_=df["open"].to_numpy(dtype=np.float64),
        high=df["high"].to_numpy(dtype=np.float64),
        low=df["low"].to_numpy(dtype=np.float64),
        close=df["close"].to_numpy(dtype=np.float64),
        atr=df["atr_14"].to_numpy(dtype=np.float64),
        swing_high=df["dc_high"].to_numpy(dtype=np.float64),
        swing_low=df["dc_low"].to_numpy(dtype=np.float64),
        test_idx=test_idx,
        capital=capital,
        risk_per_trade=0.005,
        min_rr=0.5,
        timeout_bars=24,
        timestamps=df.index.to_numpy(),
        save_csv=True,
        csv_path=RESULTS_DIR / "trade_log.csv",
        print_first_n=20,
    )

    cap = execution["final_equity"]
    equity = execution["equity_curve"]
    wins = execution["wins"]
    losses = execution["losses"]
    trades = execution["total_trades"]
    gw = sum(t["pnl"] for t in execution["trades"] if t["pnl"] > 0)
    gl = sum(abs(t["pnl"]) for t in execution["trades"] if t["pnl"] < 0)
    test_bars = len(test_idx)
    test_days = test_bars / 1440
    sharpe = 0.0

    if test_days > 2:
        test_equity = equity[test_idx]
        n_test_days = len(test_equity) // 1440
        if n_test_days > 2:
            dl = np.array([
                np.diff(
                    np.log(
                        test_equity[i * 1440:(i + 1) * 1440] + 1e-10
                    )
                ).sum()
                for i in range(n_test_days)
            ])
            sharpe = float(
                dl.mean() / (dl.std() + 1e-10) * np.sqrt(365)
            )
        else:
            sharpe = 0.0

    # Max drawdown from full equity curve
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / (peak + 1e-10) * 100
    max_dd = float(drawdown.min())

    total_ret = (cap - capital) / capital * 100
    trades = wins + losses
    win_rate = wins / trades * 100 if trades > 0 else 0.0
    calmar = total_ret / abs(max_dd) if abs(max_dd) > 0.01 else 0.0
    pf = gw / (gl + 1e-10)

    best_r = dict(
        sharpe=round(sharpe, 3),
        ret=round(total_ret, 2),
        max_dd=round(max_dd, 2),
        calmar=round(calmar, 3),
        pf=round(pf, 3),
        trades=trades,
        win_rate=round(win_rate, 1),
        final=round(cap, 2),
        equity=equity,
        signals=sig,
        long_thresh=best_long,
        short_thresh=best_short,
    )

    print(
        f"  {GREEN('[OK]')} FINAL TEST (unseen): "
        f"Ret={best_r['ret']:+.2f}%  "
        f"Sharpe={best_r['sharpe']:+.3f}  "
        f"MaxDD={best_r['max_dd']:.2f}%  "
        f"PF={best_r['pf']:.3f}  "
        f"Trades={best_r['trades']:,}  "
        f"WR={best_r.get('win_rate', best_r.get('wr', 0.0)):.1f}%  "
        f"[L>{best_long} S<{best_short}]"
    )

    # -- Save best params -----------------------------------------------------
    CONFIG_DIR.mkdir(exist_ok=True)
    params = {
        'model':         'XGBoost+LightGBM+LSTM Ensemble',
        'forward_bars':  FORWARD_BARS,
        'threshold_pct': 'dynamic_atr',
        'long_thresh':   best_r['long_thresh'],
        'short_thresh':  best_r['short_thresh'],
        'top_features':  top_features,
        'lstm_features': top_features[:20],
        'lstm_seq_len':  30,
        'results':       {k: v for k, v in best_r.items()
                          if k not in ('equity', 'signals')},
    }
    with open(CONFIG_DIR / 'ml_best_params.json', 'w') as f:
        json.dump(params, f, indent=2)

    # -- Save trained models to disk (for live trading) -----------------------
    try:
        import joblib
        MODELS_DIR = ROOT / 'models'
        MODELS_DIR.mkdir(exist_ok=True)

        # Re-train XGB + LGB on the LAST fold (most recent 30 days)
        # This is the model that will be used for live trading
        last_tr_start = folds[-1][0]; last_tr_end = folds[-1][1]
        tr_mask_live = valid_mask[last_tr_start:last_tr_end]
        tr_idx_live  = np.arange(last_tr_start, last_tr_end)[tr_mask_live]
        X_live       = X_sel[tr_idx_live]
        y_live       = encode_labels(labels[tr_idx_live])
        scaler_live  = RobustScaler()
        X_live_s     = scaler_live.fit_transform(X_live)
        flat_frac_l  = (y_live == 1).mean()
        w_flat_l     = 1.0 / (flat_frac_l + 1e-10)
        w_dir_l      = 1.0 / ((1 - flat_frac_l) / 2 + 1e-10)
        sw_live      = np.where(y_live == 1, w_flat_l, w_dir_l)

        xgb_live = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            objective='multi:softprob', num_class=3,
            eval_metric='mlogloss', random_state=42, verbosity=0,
            use_label_encoder=False,
            tree_method='hist', device=_xgb_device,
        )
        xgb_live.fit(X_live_s, y_live, sample_weight=sw_live)

        lgb_live = lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
            class_weight='balanced', random_state=42, verbose=-1,
            device='gpu' if _ml_device == 'cuda' else 'cpu',
        )
        lgb_live.fit(X_live_s, y_live, sample_weight=sw_live)

        # Bundle: models + scaler + feature list
        bundle = {
            'xgb':          xgb_live,
            'lgb':          lgb_live,
            'scaler':       scaler_live,
            'top_features': top_features,
            'lstm_features': top_features[:20],
            'lstm_seq_len': 30,
            'long_thresh':  best_r['long_thresh'],
            'short_thresh': best_r['short_thresh'],
            'lstm_state':   None,   # filled below if LSTM available
            'trained_at':   datetime.now().isoformat(),
            'pair':         pair,
        }

        # Save LSTM state dict if available
        if LSTM_AVAILABLE:
            try:
                import torch
                import torch.nn as nn
                DEVICE_LIVE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                SEQ_LEN_L   = 30
                LSTM_FEATS_L = 20
                lstm_feats_l = top_features[:LSTM_FEATS_L]
                X_lstm_l     = X_sel[:, :LSTM_FEATS_L].astype(np.float64)
                X_lstm_l     = np.nan_to_num(X_lstm_l, nan=0.0, posinf=0.0, neginf=0.0)

                class _LSTMAttn(nn.Module):
                    def __init__(self, input_size, hidden=64, n_layers=2, n_classes=3):
                        super().__init__()
                        self.lstm = nn.LSTM(input_size, hidden, n_layers,
                                            batch_first=True, dropout=0.2)
                        self.attn = nn.Linear(hidden, 1)
                        self.fc   = nn.Sequential(
                            nn.Linear(hidden, 32), nn.ReLU(),
                            nn.Dropout(0.2), nn.Linear(32, n_classes)
                        )
                    def forward(self, x):
                        out, _ = self.lstm(x)
                        attn_w = torch.softmax(self.attn(out), dim=1)
                        ctx    = (out * attn_w).sum(dim=1)
                        return self.fc(ctx)

                from torch.utils.data import DataLoader, TensorDataset
                tr_seqs_l, tr_labs_l = [], []
                for i in range(last_tr_start + SEQ_LEN_L, last_tr_end):
                    if not valid_mask[i]: continue
                    tr_seqs_l.append(X_lstm_l[i-SEQ_LEN_L:i])
                    tr_labs_l.append(encode_labels(labels[i]))

                if len(tr_seqs_l) >= 100:
                    X_tr_tl = torch.tensor(np.array(tr_seqs_l), dtype=torch.float32).to(DEVICE_LIVE)
                    y_tr_tl = torch.tensor(tr_labs_l, dtype=torch.long).to(DEVICE_LIVE)
                    fm_l    = X_tr_tl.mean(dim=(0,1), keepdim=True)
                    fs_l    = X_tr_tl.std(dim=(0,1), keepdim=True) + 1e-8
                    X_tr_tl = (X_tr_tl - fm_l) / fs_l
                    ds_l    = TensorDataset(X_tr_tl, y_tr_tl)
                    ld_l    = DataLoader(ds_l, batch_size=512, shuffle=True)
                    lstm_model_live = _LSTMAttn(LSTM_FEATS_L).to(DEVICE_LIVE)
                    opt_l   = torch.optim.Adam(lstm_model_live.parameters(), lr=1e-3)
                    loss_fn_l = nn.CrossEntropyLoss()
                    for _ in range(5):
                        for xb, yb in ld_l:
                            opt_l.zero_grad()
                            loss_fn_l(lstm_model_live(xb), yb).backward()
                            opt_l.step()
                    bundle['lstm_state']    = lstm_model_live.cpu().state_dict()
                    bundle['lstm_feat_mean'] = fm_l.cpu().numpy()
                    bundle['lstm_feat_std']  = fs_l.cpu().numpy()
            except Exception as _e:
                import traceback
                print(f"  {YELLOW('вљ ')} LSTM save skipped: {_e}")
                traceback.print_exc()

            if best_r.get("sharpe", -1.0) <= 0.0 or best_r.get("pf", 0.0) < 1.0:
                print("  [!] SAFETY GATE: Sharpe <= 0 or PF < 1.0. Model rejected, bundle NOT saved.")
            else:
                joblib.dump(bundle, MODELS_DIR / 'ml_live_bundle.pkl')
                bundle_size = (MODELS_DIR / 'ml_live_bundle.pkl').stat().st_size / 1e6
                print(f"  [OK] Live model bundle saved -> models/ml_live_bundle.pkl  ({bundle_size:.1f} MB)")
                print(f"  Contains: XGBoost + LightGBM + scaler + features + thresholds" + (" + LSTM" if bundle.get('lstm_state') is not None else ""))

    except Exception as _e:
        print(f"  {YELLOW('вљ ')} Model save failed: {_e}")

    # -- Print results table --------------------------------------------------
    col = GREEN if best_r['ret'] >= 0 else RED
    ret_str    = f"{best_r['ret']:>+.2f}%"
    sharpe_str = f"{best_r['sharpe']:>22.4f}"
    calmar_str = f"{best_r['calmar']:>22.4f}"
    pf_str     = f"{best_r['pf']:>22.3f}x"
    dd_str     = f"{best_r['max_dd']:.2f}%"
    wr_str     = f"{best_r.get('win_rate', best_r.get('wr', 0.0)):.1f}%"
    lt_str     = f"{best_r['long_thresh']:>22}"
    st_str     = f"{best_r['short_thresh']:>22}"
    print()
    print(f"  {'-'*56}")
    print(f"  ML ENSEMBLE BACKTEST RESULTS  |  {BOLD(pair)}")
    print(f"  {'-'*56}")
    print(f"  Initial Capital   {WHITE('$'):>4}{capital:>19,.2f}")
    print(f"  Final Value       {col('$'):>4}{best_r['final']:>19,.2f}")
    print(f"  Total Return      {col(ret_str):>23}")
    print(f"  Sharpe Ratio      {sharpe_str}")
    print(f"  Calmar Ratio      {calmar_str}")
    print(f"  Profit Factor     {pf_str}")
    print(f"  Max Drawdown      {RED(dd_str):>23}")
    print(f"  Total Trades      {best_r['trades']:>22,}")
    print(f"  Win Rate          {wr_str:>22}")
    print(f"  Long Threshold    {lt_str}")
    print(f"  Short Threshold   {st_str}")
    print(f"  Data Source       {'Binance Vision + Hyperliquid':>22}")
    print(f"  Model             {'XGB 40% + LGB 40% + LSTM 20%':>22}")
    print(f"  {'-'*56}")

    # -- Save chart -----------------------------------------------------------
    _save_ml_chart(df, best_r, pair, capital, ens_score, oos_mask, days)

    return best_r


# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
# CHART вЂ” ML Ensemble
# в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

def _save_ml_chart(df, best_r, pair, capital, ens_score, oos_mask, days=None):
    """Save a dark-mode ML ensemble performance chart to results/."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        equity = best_r['equity']
        sig    = best_r['signals']
        n      = len(equity)
        dates  = df.index
        c      = df['close'].to_numpy(np.float64)
        dd     = (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity) * 100
        rets   = np.diff(equity) / (equity[:-1] + 1e-10)

        fig = plt.figure(figsize=(18, 12), facecolor='#0d1117')
        fig.suptitle(
            f"AIQuant  |  ML Ensemble (XGB+LGB+LSTM)  |  {pair}  |  {days}-Day Backtest\n"
            f"Sharpe {best_r['sharpe']:+.3f}  |  Return {best_r['ret']:+.1f}%  |  "
            f"MaxDD {best_r['max_dd']:.1f}%  |  Calmar {best_r['calmar']:.3f}  |  "
            f"{best_r['trades']:,} trades  |  {best_r.get('win_rate', best_r.get('wr', 0.0)):.1f}% win rate  |  "
            f"Profit Factor {best_r['pf']:.2f}x",
            color='white', fontsize=11, fontweight='bold', y=0.99
        )

        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

        def _ax(subplot):
            ax = fig.add_subplot(subplot)
            ax.set_facecolor('#161b22')
            ax.tick_params(colors='#8b949e', labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor('#30363d')
            return ax

        # 1. Equity curve (full width)
        ax1 = _ax(gs[0, :])
        ax1.plot(dates, equity, color='#00d4aa', linewidth=0.9)
        ax1.fill_between(dates, equity, capital,
                         where=equity >= capital, alpha=0.15, color='#00d4aa')
        ax1.fill_between(dates, equity, capital,
                         where=equity < capital, alpha=0.15, color='#ff4444')
        ax1.axhline(capital, color='#555', linestyle='--', linewidth=0.8, alpha=0.6)
        ax1.set_ylabel('Portfolio ($)', color='white', fontsize=9)
        ax1.set_title('Equity Curve (Out-of-Sample Only)', color='white', fontsize=9)

        # 2. Drawdown (full width)
        ax2 = _ax(gs[1, :])
        ax2.fill_between(dates, dd, 0, color='#ff4444', alpha=0.7)
        ax2.set_ylabel('Drawdown (%)', color='white', fontsize=9)
        ax2.set_title('Drawdown', color='white', fontsize=9)

        # 3. BTC price + signals
        step = max(1, n // 2000)
        ax3  = _ax(gs[2, 0])
        ax3.plot(dates[::step], c[::step], color='#f0a500', linewidth=0.6, alpha=0.8)
        li = np.where(sig == 1)[0][::step]
        si = np.where(sig == -1)[0][::step]
        if len(li): ax3.scatter(dates[li], c[li], c='#00d4aa', s=1.5, alpha=0.5, zorder=3)
        if len(si): ax3.scatter(dates[si], c[si], c='#ff4444', s=1.5, alpha=0.5, zorder=3)
        ax3.set_title('BTC Price + ML Signals', color='white', fontsize=9)

        # 4. Return distribution
        ax4 = _ax(gs[2, 1])
        cr  = rets[np.isfinite(rets) & (np.abs(rets) < 0.05)] * 100
        ax4.hist(cr, bins=80, color='#58a6ff', alpha=0.85, edgecolor='none')
        ax4.axvline(0, color='white', linewidth=0.8, linestyle='--')
        if len(cr) > 0:
            ax4.axvline(cr.mean(), color='#00d4aa', linewidth=1.0, linestyle='--', alpha=0.8)
        ax4.set_title('Return Distribution', color='white', fontsize=9)
        ax4.set_xlabel('Return (%)', color='#8b949e', fontsize=8)

        # 5. Ensemble score distribution
        ax5     = _ax(gs[2, 2])
        ens_oos = ens_score[oos_mask]
        ax5.hist(ens_oos, bins=80, color='#bc8cff', alpha=0.85, edgecolor='none')
        ax5.axvline(best_r['long_thresh'],  color='#00d4aa', linewidth=1.2,
                    linestyle='--', label=f'Long >{best_r["long_thresh"]}')
        ax5.axvline(best_r['short_thresh'], color='#ff4444', linewidth=1.2,
                    linestyle='--', label=f'Short <{best_r["short_thresh"]}')
        ax5.axvline(0, color='white', linewidth=0.8)
        ax5.set_title('Ensemble Score Distribution', color='white', fontsize=9)
        ax5.set_xlabel('Long-Short Score', color='#8b949e', fontsize=8)
        ax5.legend(facecolor='#161b22', labelcolor='white', fontsize=7)

        out = RESULTS_DIR / 'backtest_results.png'
        plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        plt.close(fig)
        print(f"\n  {GREEN('[OK]')} Chart saved -> {out}")

        # Try to open on desktop environments
        if platform.system() == 'Darwin':
            os.system(f'open "{out}"')
        elif platform.system() == 'Linux' and os.environ.get('DISPLAY'):
            os.system(f'xdg-open "{out}" 2>/dev/null &')

    except Exception as e:
        print(f"  {YELLOW('вљ ')}  Chart generation failed: {e}")