@echo off

echo TEST 0.0008
set LABEL_THRESHOLD=0.0008
python your_backtest_runner.py

echo TEST 0.0010
set LABEL_THRESHOLD=0.0010
python your_backtest_runner.py

echo TEST 0.0012
set LABEL_THRESHOLD=0.0012
python your_backtest_runner.py

echo TEST 0.0015
set LABEL_THRESHOLD=0.0015
python your_backtest_runner.py