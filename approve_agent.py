"""
QVEX v10.7 — Утилита разовой авторизации Agent Wallet на Hyperliquid L1.
Считывает мастер-ключ и адрес агента из .env и выполняет транзакцию EIP-712.
"""
import os
import sys
from dotenv import load_dotenv
from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

load_dotenv()

def main():
    master_key = os.getenv("SECRET_KEY")
    agent_address = os.getenv("AGENT_ADDRESS")
    is_testnet = os.getenv("TESTNET", "True").lower() == "true"
    base_url = constants.TESTNET_API_URL if is_testnet else constants.MAINNET_API_URL

    if not master_key or not agent_address:
        print("[!] Ошибка: В файле .env отсутствуют SECRET_KEY или AGENT_ADDRESS.")
        return

    try:
        master_account = Account.from_key(master_key)
    except Exception as parse_err:
        print(f"[!] Ошибка разбора SECRET_KEY из .env: {parse_err}")
        return

    print("=" * 70)
    print("🔑 АВТОРИЗАЦИЯ АГЕНТСКОГО КОШЕЛЬКА НА HYPERLIQUID L1")
    print(f"• Сеть:          {'TESTNET' if is_testnet else 'MAINNET'}")
    print(f"• Мастер-счет:   {master_account.address}")
    print(f"• Агент:         {agent_address}")
    print("=" * 70)

    exchange = Exchange(master_account, base_url)
    try:
        result = exchange.approve_agent(agent_address)
        # Строгая валидация статуса ответа L1
        status_dict = result[0] if isinstance(result, tuple) else result
        if isinstance(status_dict, dict) and status_dict.get("status") == "err":
            err_msg = status_dict.get("response", "Неизвестная ошибка")
            print(f"❌ БИРЖА ОТКЛОНИЛА ДЕЙСТВИЕ: {err_msg}")
            if "Must deposit" in str(err_msg):
                print("💡 ДЕЙСТВИЕ: Зайдите на app.hyperliquid-testnet.xyz и запросите тестовые USDC через Faucet!")
            return

        print(f"✔ Ответ биржи L1: {result}")
        print("✔ Агент успешно наделен торговыми правами без права вывода средств!")
    except Exception as e:
        print(f"❌ Ошибка авторизации агента: {e}")

if __name__ == "__main__":
    main()
