#!/usr/bin/env python3
"""
Institutional Long Pullback with Confirmation Bar (BTC 1H).
Targets: Max DD < 8%, Daily DD < 4%, Capital $10,000.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

DATA_PATH = Path("data/BTCUSDT_15m.csv")
if not DATA_PATH.exists():
    print("[-] Ошибка: Файл data/BTCUSDT_15m.csv не найден.")
    sys.exit(1)

print(f"[*] Чтение данных: {DATA_PATH}")
raw_df = pd.read_csv(DATA_PATH)
raw_df.columns = [c.lower() for c in raw_df.columns]

# 1. Агрегация 15m -> 1H
n_raw = len(raw_df)
bar_group = np.arange(n_raw) // 4

df_1h = pd.DataFrame({
    'open': raw_df['open'].groupby(bar_group).first(),
    'high': raw_df['high'].groupby(bar_group).max(),
    'low': raw_df['low'].groupby(bar_group).min(),
    'close': raw_df['close'].groupby(bar_group).last(),
    'volume': raw_df['volume'].groupby(bar_group).sum(),
}).reset_index(drop=True)

end_time = pd.Timestamp.now().floor('h')
df_1h['timestamp'] = pd.date_range(end=end_time, periods=len(df_1h), freq='1h')

# 2. Стационарные признаки
o = df_1h['open']
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
features['timestamp'] = df_1h['timestamp']
features['open'] = df_1h['open']
features['high'] = df_1h['high']
features['low'] = df_1h['low']
features['close'] = df_1h['close']
features['volume'] = df_1h['volume']
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
print(f"[+] Сформировано {n_clean:,} 1H баров.")

# 3. Primary Model: Откат + Свечное подтверждение (Bullish Bar Confirmation)
o_arr = clean_df['open'].to_numpy(dtype=np.float64)
c_arr = clean_df['close'].to_numpy(dtype=np.float64)
h_arr = clean_df['high'].to_numpy(dtype=np.float64)
l_arr = clean_df['low'].to_numpy(dtype=np.float64)
atr_arr = clean_df['atr_fast'].to_numpy(dtype=np.float64)
ts_arr = pd.to_datetime(clean_df['timestamp']).to_numpy()

ema_25 = pd.Series(c_arr).ewm(span=25, adjust=False).mean().to_numpy()
ema_100 = pd.Series(c_arr).ewm(span=100, adjust=False).mean().to_numpy()
rsi_arr = (clean_df['rsi_norm'].to_numpy() * 50.0) + 50.0

# Условия: Аптренд + Откат + Бар закрылся в плюс (C > O) с закрытием в верхней половине
macro_bull = c_arr > ema_100
pullback_state = (rsi_arr < 45.0) | (l_arr <= ema_25)
bull_bar = (c_arr > o_arr) & (c_arr >= (h_arr + l_arr) / 2.0)
primary_signals = macro_bull & pullback_state & bull_bar

# Возвращаем проверенные барьеры Теста А
SL_MULT = 1.20
TP_MULT = 1.80   # RR = 1.50
TIMEOUT_BARS = 20

# 4. Разметка Meta-Labels
meta_labels = np.full(n_clean, np.nan)
for i in range(n_clean - TIMEOUT_BARS):
    if not primary_signals[i]:
        continue
    entry = c_arr[i]
    cur_atr = atr_arr[i]
    sl_price = entry - (cur_atr * SL_MULT)
    tp_price = entry + (cur_atr * TP_MULT)
    
    success = 0
    for k in range(1, TIMEOUT_BARS + 1):
        if l_arr[i + k] <= sl_price:
            success = 0
            break
        if h_arr[i + k] >= tp_price:
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
print(f"[+] Отобрано подтвержденных лонг-сетапов: {len(setup_indices):,}")
print(f"[+] Базовый винрейт эвристики: {(y_meta == 1).mean() * 100:.2f}%")

# 5. Обучение Meta-XGBoost
TRAIN_SIZE = int(len(setup_indices) * 0.65)
X_train, y_train = X_meta[:TRAIN_SIZE], y_meta[:TRAIN_SIZE]
X_test,  y_test  = X_meta[TRAIN_SIZE:], y_meta[TRAIN_SIZE:]

pos_count = sum(y_train)
scale_pos = (len(y_train) - pos_count) / pos_count

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

# 6. Бэктест OOS с PropGuard
INITIAL_CAPITAL = 10000.0
equity = INITIAL_CAPITAL
equity_curve = [equity]
trades = []

FEE_PCT = 0.0012
RISK_PCT = 0.0060
CONF_THRESHOLD = 0.55

test_setup_bars = setup_indices[TRAIN_SIZE:]
setup_map = {bar: prob for bar, prob in zip(test_setup_bars, test_probs)}

oos_start = test_setup_bars[0]
in_pos = False
entry_bar = 0
entry_price = 0.0
pos_size_usd = 0.0
cur_day = None
daily_pnl = 0.0
day_locked = False

for i in range(oos_start, n_clean - 1):
    bar_date = pd.Timestamp(ts_arr[i]).date()
    if bar_date != cur_day:
        cur_day = bar_date
        daily_pnl = 0.0
        day_locked = False
        
    if in_pos:
        dur = i - entry_bar
        cur_h = h_arr[i]
        cur_l = l_arr[i]
        cur_c = c_arr[i]
        
        sl_price = entry_price - (atr_arr[entry_bar] * SL_MULT)
        tp_price = entry_price + (atr_arr[entry_bar] * TP_MULT)
        
        exit_trade = False
        exit_p = cur_c
        reason = ""
        
        if cur_l <= sl_price:
            exit_p = sl_price
            reason = "SL"
            exit_trade = True
        elif cur_h >= tp_price:
            exit_p = tp_price
            reason = "TP"
            exit_trade = True
        elif dur >= TIMEOUT_BARS:
            exit_p = cur_c
            reason = "TIMEOUT"
            exit_trade = True
                
        if exit_trade:
            net_ret = (exit_p / entry_price - 1.0) - FEE_PCT
            pnl = pos_size_usd * net_ret
            equity += pnl
            daily_pnl += pnl
            trades.append({
                "entry_time": ts_arr[entry_bar],
                "exit_time": ts_arr[i],
                "side": "LONG",
                "entry": entry_price,
                "exit": exit_p,
                "pnl": pnl,
                "reason": reason
            })
            in_pos = False
            
            if daily_pnl <= -200.0:
                day_locked = True
                
    equity_curve.append(equity)
    
    if not in_pos and not day_locked and i in setup_map:
        p_win = setup_map[i]
        if p_win >= CONF_THRESHOLD:
            entry_price = c_arr[i]
            entry_bar = i
            in_pos = True
            
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
print("  ИТОГИ ПОДТВЕРЖДЕННОЙ СИСТЕМЫ (PROP READY)")
print("=" * 65)
print(f"Начальный баланс     : ${INITIAL_CAPITAL:,.2f}")
print(f"Конечный баланс      : ${equity:,.2f}")
print(f"Чистый PnL           : {total_ret:+.2f}%")
print(f"Максимальная просадка: {max_dd:.2f}% (Норматив: < 8.0%)")
print(f"Всего сделок         : {len(df_res)} (~{len(df_res)/9:.1f} сделок/мес)")
print(f"Win Rate             : {wr:.1f}%")
print(f"Profit Factor        : {pf:.3f}")
if len(df_res) > 0:
    print("\nПричины выходов:")
    print(df_res['reason'].value_counts())
print("=" * 65)

# 8. Сохранение логов и графика
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
df_res.to_csv(results_dir / "prop_trades_log.csv", index=False)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(eq_arr, label="Equity Curve ($)", color="#1f77b4", lw=1.5)
plt.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.7)
plt.title("Prop Challenge Strategy Equity (1H Bull Confirmation)")
plt.ylabel("Balance ($)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(dd * 100.0, label="Drawdown (%)", color="#d62728", lw=1.2)
plt.axhline(-8.0, color="black", linestyle="--", label="Prop Limit (-8%)")
plt.ylabel("Drawdown (%)")
plt.xlabel("Hours")
plt.grid(True, alpha=0.3)
plt.legend()

chart_path = results_dir / "prop_backtest_chart.png"
plt.tight_layout()
plt.savefig(chart_path, dpi=150)
plt.close()
print(f"[+] График сохранен в: {chart_path}")
print(f"[+] Лог сохранен в: {results_dir / 'prop_trades_log.csv'}")