#!/usr/bin/env python3
"""
Canonical CLI Entry Point for HL-Swing-Trend Engine.
Usage:
    python run.py testnet     - Запуск боевого сервиса на Hyperliquid Testnet
    python run.py mainnet     - Запуск боевого сервиса на Hyperliquid Mainnet
    python run.py backtest    - Запуск реалистичного бэктеста (с фандингом и проскальзыванием)
    python run.py stress      - Запуск полного комплекса квантовых стресс-тестов
"""

import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Hyperliquid 4H Swing Trend Runner")
    parser.add_argument("mode", choices=["testnet", "mainnet", "backtest", "stress"], help="Режим запуска")
    args = parser.parse_args()

    if args.mode in ["testnet", "mainnet"]:
        import bot_config as config
        config.IS_TESTNET = (args.mode == "testnet")
        print(f"[*] Запуск бота в режиме: {args.mode.upper()}...")
        from hl_swing_bot import HyperliquidSwingBot
        bot = HyperliquidSwingBot()
        bot.start()

    elif args.mode == "backtest":
        print("[*] Запуск симулятора с учетом реальных рыночных издержек...")
        subprocess.run([sys.executable, "run_hardened_pullback.py"])

    elif args.mode == "stress":
        print("[*] Запуск стресс-тестов Монте-Карло и сетки устойчивости...")
        subprocess.run([sys.executable, "run_full_stress_suite.py"])

if __name__ == "__main__":
    main()
