import pandas as pd, numpy as np
from aiquant.strategy.backtest_execution import simulate_trade_plan

df = pd.read_parquet('data/raw/BTCUSDT_1m.parquet').iloc[:10000]
n = len(df)
tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
atr = tr.rolling(14).mean().bfill().values
sig = np.zeros(n); sig[::45] = 1; sig[22::45] = -1
res = simulate_trade_plan(signal=sig, open_=df['open'].values, high=df['high'].values, low=df['low'].values, close=df['close'].values, atr=atr, swing_high=df['high'].rolling(20).max().values, swing_low=df['low'].rolling(20).min().values, test_idx=np.arange(50, n-100), timeout_bars=60, min_rr=1.1)

trades = res['trades']
timeout_trades = [t for t in trades if t['exit_reason'] == 'TIMEOUT']

mfe_list, mae_list = [], []
high_arr, low_arr = df['high'].values, df['low'].values
for t in timeout_trades:
    e_idx, x_idx = t['entry_idx'], t['exit_idx']
    entry = t['entry']
    risk_dist = abs(entry - t['stop_loss'])
    if t['side'] == 'LONG':
        max_fav = high_arr[e_idx:x_idx+1].max() - entry
        max_adv = entry - low_arr[e_idx:x_idx+1].min()
    else:
        max_fav = entry - low_arr[e_idx:x_idx+1].min()
        max_adv = high_arr[e_idx:x_idx+1].max() - entry
    mfe_list.append(max_fav / risk_dist)
    mae_list.append(max_adv / risk_dist)

print('='*60)
print('ДИАГНОСТИКА TIMEOUT СДЕЛОК (MFE / MAE)')
print('='*60)
print(f'Всего выходов по TIMEOUT : {len(timeout_trades)}')
print(f'Средний MFE (макс. прибыль в R): {np.mean(mfe_list):.2f}R')
print(f'Сделок доходивших до +0.8R     : {sum(x >= 0.8 for x in mfe_list)} ({sum(x >= 0.8 for x in mfe_list)/len(mfe_list)*100:.1f}%)')
print(f'Сделок доходивших до +1.0R     : {sum(x >= 1.0 for x in mfe_list)} ({sum(x >= 1.0 for x in mfe_list)/len(mfe_list)*100:.1f}%)')
print(f'Сделок доходивших до +1.2R     : {sum(x >= 1.2 for x in mfe_list)} ({sum(x >= 1.2 for x in mfe_list)/len(mfe_list)*100:.1f}%)')
print(f'Средний MAE (макс. просадка в R): {np.mean(mae_list):.2f}R')
print('='*60)
