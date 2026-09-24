#!/usr/bin/env python3
"""
Institutional 1H BTC Engine (Dual-Tranche Proven Clean Benchmark).
50% TP1 at 1.2 ATR, 50% TP2 at 3.0 ATR. Risk 0.6%.
Hard limits: Max DD < 8.0%, Daily Loss < $200, Monoposition.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

DATA_PATH = Path("data/BTCUSDT_15m.csv")
if not DATA_PATH.exists():
    print(f"[-] Ошибка: Файл {DATA_PATH} не найден.")
    sys.exit(1)

print(f"[*] Загрузка датасета: {DATA_PATH}")
raw = pd.read_csv(DATA_PATH)
raw.columns = raw.columns.str.lower()

# 1. Агрегация 15m -> 1H
n_raw = len(raw)
bar_group = np.arange(n_raw) // 4

df_1h = pd.DataFrame({
    'open': raw['open'].groupby(bar_group).first(),
    'high': raw['high'].groupby(bar_group).max(),
    'low': raw['low'].groupby(bar_group).min(),
    'close': raw['close'].groupby(bar_group).last(),
    'volume': raw['volume'].groupby(bar_group).sum(),
    'hour': raw['hour'].groupby(bar_group).first() if 'hour' in raw.columns else np.zeros(len(raw)//4)
}).reset_index(drop=True)

df_1h['day_id'] = np.arange(len(df_1h)) // 24
n_bars = len(df_1h)
print(f"[+] Сформировано {n_bars:,} 1H баров.")

# 2. Расчет стационарных признаков
c = df_1h['close']
h = df_1h['high']
l = df_1h['low']
v = df_1h['volume']

tr1 = h - l
tr2 = (h - c.shift(1)).abs()
tr3 = (l - c.shift(1)).abs()
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

atr_fast = tr.rolling(window=6, min_periods=6).mean()
atr_slow = tr.rolling(window=48, min_periods=48).mean()
atr_norm = atr_fast / c

features = pd.DataFrame(index=df_1h.index)
features['open'] = df_1h['open']
features['high'] = df_1h['high']
features['low'] = df_1h['low']
features['close'] = df_1h['close']
features['volume'] = df_1h['volume']
features['hour'] = df_1h['hour']
features['day_id'] = df_1h['day_id']
features['atr_fast'] = atr_fast

features['log_ret_1h'] = np.log(c / c.shift(1))
features['log_ret_4h'] = np.log(c / c.shift(4))
features['log_ret_24h'] = np.log(c / c.shift(24))

ema_f = c.ewm(span=12, adjust=False).mean()
ema_s = c.ewm(span=48, adjust=False).mean()
features['trend_spread_atr'] = (ema_f - ema_s) / atr_fast
features['vol_compression'] = atr_fast / atr_slow

hl_comp = (np.log(h / l) ** 2) / (4 * np.log(2))
parkinson_vol = np.sqrt(hl_comp.rolling(window=12, min_periods=12).mean())
features['parkinson_to_atr'] = parkinson_vol / (atr_norm + 1e-8)

range_safe = (h - l).replace(0, np.nan)
features['bar_pressure'] = ((2 * c - h - l) / range_safe).fillna(0.0)

vol_mean = v.rolling(window=48, min_periods=48).mean()
vol_std = v.rolling(window=48, min_periods=48).std()
features['volume_zscore'] = (v - vol_mean) / (vol_std + 1e-8)

delta = c.diff()
gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
loss = ((-delta.where(delta < 0, 0.0))).ewm(alpha=1/14, adjust=False).mean()
rs = gain / (loss + 1e-12)
rsi = 100.0 - (100.0 / (1.0 + rs))
features['rsi_norm'] = (rsi - 50.0) / 50.0

ema_200 = c.ewm(span=200, adjust=False).mean()
features['dist_ema200_atr'] = (c - ema_200) / atr_fast

clean_df = features.dropna().reset_index(drop=True)
n_clean = len(clean_df)

# 3. Чистый базовый сетап
c_arr = clean_df['close'].to_numpy(dtype=np.float64)
o_arr = clean_df['open'].to_numpy(dtype=np.float64)
h_arr = clean_df['high'].to_numpy(dtype=np.float64)
l_arr = clean_df['low'].to_numpy(dtype=np.float64)
atr_arr = clean_df['atr_fast'].to_numpy(dtype=np.float64)
day_arr = clean_df['day_id'].to_numpy(dtype=np.int32)

ema_25 = pd.Series(c_arr).ewm(span=25, adjust=False).mean().to_numpy()
ema_100 = pd.Series(c_arr).ewm(span=100, adjust=False).mean().to_numpy()
rsi_arr = (clean_df['rsi_norm'].to_numpy() * 50.0) + 50.0

macro_bull = c_arr > ema_100
pullback = (rsi_arr < 42.0) | (l_arr <= ema_25)
primary_signals = macro_bull & pullback

SL_MULT = 1.30
TP1_MULT = 1.20   # 50% объема
TP2_MULT = 3.00   # 50% объема
TIMEOUT_BARS = 36

# 4. Разметка Meta-Labels
meta_labels = np.full(n_clean, np.nan)
for i in range(n_clean - TIMEOUT_BARS - 1):
    if not primary_signals[i]:
        continue
    entry = o_arr[i + 1]
    cur_atr = atr_arr[i]
    sl_price = entry - (cur_atr * SL_MULT)
    tp1_price = entry + (cur_atr * TP1_MULT)

    success = 0
    for k in range(1, TIMEOUT_BARS + 1):
        idx = i + k
        if l_arr[idx] <= sl_price:
            success = 0
            break
        if h_arr[idx] >= tp1_price:
            success = 1
            break
    meta_labels[i] = success

setup_indices = np.where(~np.isnan(meta_labels))[0]
y_meta = meta_labels[setup_indices].astype(int)

X_cols = [
    'log_ret_1h', 'log_ret_4h', 'log_ret_24h', 
    'trend_spread_atr', 'vol_compression', 'parkinson_to_atr', 
    'bar_pressure', 'volume_zscore', 'rsi_norm', 'dist_ema200_atr'
]
X_meta = clean_df.loc[setup_indices, X_cols].to_numpy(dtype=np.float32)

print(f"[+] Всего сетапов: {len(setup_indices):,}")

# 5. Обучение Meta-XGBoost с зазором
TRAIN_RATIO = 0.65
split_idx = int(len(setup_indices) * TRAIN_RATIO)
train_setup_indices = setup_indices[:split_idx]

last_train_bar = train_setup_indices[-1]
test_start_bar = last_train_bar + TIMEOUT_BARS + 1
test_setup_indices = setup_indices[setup_indices >= test_start_bar]

X_train = clean_df.loc[train_setup_indices, X_cols].to_numpy(dtype=np.float32)
y_train = meta_labels[train_setup_indices].astype(int)
X_test  = clean_df.loc[test_setup_indices, X_cols].to_numpy(dtype=np.float32)

pos_count = sum(y_train)
scale_pos = (len(y_train) - pos_count) / max(pos_count, 1)

meta_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    random_state=42,
    tree_method='hist'
)
meta_clf.fit(X_train, y_train)
test_probs = meta_clf.predict_proba(X_test)[:, 1]

setup_prob_map = {bar: prob for bar, prob in zip(test_setup_indices, test_probs)}

# 6. Бэктест OOS (Clean Proven Settings)
INITIAL_CAPITAL = 10000.0
equity = INITIAL_CAPITAL
peak_equity = equity
equity_curve = [equity]
trades = []

FEE_PCT = 0.0012
RISK_PCT = 0.0060        # 0.6% ($60 риска)
CONF_THRESHOLD = 0.55

in_pos = False
entry_bar = 0
entry_price = 0.0
pos_size_usd = 0.0
sl_price = 0.0
tp1_price = 0.0
tp2_price = 0.0
tp1_hit = False

cur_day_id = None
daily_realized_pnl = 0.0
day_locked = False
system_halted = False

oos_start = test_setup_indices[0]

for i in range(oos_start, n_clean - 1):
    bar_day = day_arr[i]

    if bar_day != cur_day_id:
        cur_day_id = bar_day
        daily_realized_pnl = 0.0
        day_locked = False

    if system_halted:
        equity_curve.append(equity)
        continue

    if in_pos:
        dur = i - entry_bar
        cur_h = h_arr[i]
        cur_l = l_arr[i]
        cur_c = c_arr[i]

        # 1. Первый транш (50%)
        if not tp1_hit and cur_h >= tp1_price:
            tp1_hit = True
            net_ret1 = (tp1_price / entry_price - 1.0) - FEE_PCT
            pnl1 = (pos_size_usd * 0.5) * net_ret1
            equity += pnl1
            daily_realized_pnl += pnl1

        exit_remaining = False
        exit_p = cur_c
        reason = ""

        if cur_l <= sl_price:
            exit_p = sl_price
            reason = "SL_AFTER_TP1" if tp1_hit else "SL_FULL"
            exit_remaining = True
        elif tp1_hit and cur_h >= tp2_price:
            exit_p = tp2_price
            reason = "TP2_FULL_WIN"
            exit_remaining = True
        elif dur >= TIMEOUT_BARS:
            exit_p = cur_c
            reason = "TIMEOUT"
            exit_remaining = True

        if exit_remaining:
            rem_fraction = 0.5 if tp1_hit else 1.0
            net_ret2 = (exit_p / entry_price - 1.0) - FEE_PCT
            pnl2 = (pos_size_usd * rem_fraction) * net_ret2
            equity += pnl2
            daily_realized_pnl += pnl2

            total_trade_pnl = (pnl1 + pnl2) if tp1_hit else pnl2
            trades.append({
                "entry_bar": entry_bar,
                "exit_bar": i,
                "entry": entry_price,
                "exit": exit_p,
                "pnl": total_trade_pnl,
                "reason": reason
            })
            in_pos = False
            tp1_hit = False

            if daily_realized_pnl <= -200.0:
                day_locked = True

    peak_equity = max(peak_equity, equity)
    drawdown = (peak_equity - equity) / peak_equity
    if drawdown >= 0.08:
        print(f"[!] АВАРИЙНЫЙ ХАРД-СТОП: Достигнут лимит просадки {drawdown*100:.2f}%! Торги остановлены.")
        system_halted = True

    equity_curve.append(equity)

    if not in_pos and not day_locked and not system_halted and i in setup_prob_map:
        if setup_prob_map[i] >= CONF_THRESHOLD:
            next_idx = i + 1
            if next_idx < n_clean:
                entry_price = o_arr[next_idx]
                entry_bar = next_idx
                in_pos = True
                tp1_hit = False

                sl_price = entry_price - (atr_arr[i] * SL_MULT)
                tp1_price = entry_price + (atr_arr[i] * TP1_MULT)
                tp2_price = entry_price + (atr_arr[i] * TP2_MULT)

                stop_dist_pct = (atr_arr[i] * SL_MULT) / entry_price
                pos_size_usd = min((equity * RISK_PCT) / stop_dist_pct, equity * 1.5)

# 7. Итоговая аналитика
df_res = pd.DataFrame(trades)
eq_arr = np.array(equity_curve)
peak = np.maximum.accumulate(eq_arr)
dd = (eq_arr - peak) / peak
max_dd = dd.min() * 100.0
total_ret = ((equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100.0

win_t = df_res[df_res['pnl'] > 0]
lose_t = df_res[df_res['pnl'] < 0]
wr = (len(win_t) / len(df_res)) * 100.0 if len(df_res) > 0 else 0.0
pf = (win_t['pnl'].sum() / abs(lose_t['pnl'].sum())) if len(lose_t) > 0 else 0.0

print("=" * 65)
print("  ИТОГИ: ВОССТАНОВЛЕННЫЙ ЭТАЛОН DUAL-TRANCHE")
print("=" * 65)
print(f"Начальный баланс     : ${INITIAL_CAPITAL:,.2f}")
print(f"Конечный баланс      : ${equity:,.2f}")
print(f"Чистый PnL           : {total_ret:+.2f}%")
print(f"Максимальная просадка: {max_dd:.2f}% (Лимит: < 8.0%)")
print(f"Всего сделок         : {len(df_res)} (~{len(df_res)/9:.1f} сделок/мес)")
print(f"Win Rate             : {wr:.1f}%")
print(f"Profit Factor        : {pf:.3f}")
if len(df_res) > 0:
    print("\nПричины выходов:")
    print(df_res['reason'].value_counts())
print("=" * 65)

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
df_res.to_csv(results_dir / "prop_trades_log.csv", index=False)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(eq_arr, label="Equity ($)", color="#1f77b4", lw=1.5)
plt.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.7)
plt.title("Prop Challenge Pure Baseline Equity (1H BTC)")
plt.ylabel("Balance ($)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(dd * 100.0, label="Drawdown (%)", color="#d62728", lw=1.2)
plt.axhline(-8.0, color="black", linestyle="--", label="Prop Limit (-8%)")
plt.ylabel("Drawdown (%)")
plt.xlabel("Bars (1H)")
plt.grid(True, alpha=0.3)
plt.legend()

chart_path = results_dir / "prop_backtest_chart.png"
plt.tight_layout()
plt.savefig(chart_path, dpi=150)
plt.close()
print(f"[+] Результаты сохранены в results/")
