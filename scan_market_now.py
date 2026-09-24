#!/usr/bin/env python3
import time
from hyperliquid.info import Info
from hyperliquid.utils import constants
import bot_config as config
from cs_ranker import CrossSectionalRanker

print("=" * 78)
print("  СКАНИРОВАНИЕ ВСЕЛЕННОЙ АКТИВОВ: КРОСС-СЕКЦИОННЫЙ РАНЖИРОВЩИК")
print("=" * 78)

base_url = constants.TESTNET_API_URL if config.IS_TESTNET else constants.MAINNET_API_URL
info = Info(base_url, skip_ws=True)

ranker = CrossSectionalRanker(info, config.TARGET_COINS)

print(f"[*] Сбор 72H доходности и L1-фандинга по {len(config.TARGET_COINS)} монетам...")
t0 = time.time()
df = ranker.rank_universe(max_funding_apr=35.0)
dt = time.time() - t0

if df.empty:
    print("[-] Ошибка: не удалось получить данные с ноды биржи.")
else:
    print(f"[✓] Анализ завершен за {dt:.1f} сек!\n")
    print(f"{'РАНГ':<5} | {'АКТИВ':<7} | {'72H ДОХОД':<11} | {'RS К BTC':<11} | {'ФАНДИНГ APR':<13} | {'СТАТУС'}")
    print("-" * 78)
    for _, r in df.iterrows():
        print(f"#{r['rank']:<4} | {r['coin']:<7} | {r['ret_72h_pct']:+8.2f}% | {r['rs_to_btc_pct']:+8.2f}% | {r['funding_apr']:+9.1f}% APR | {r['status']}")

    print("-" * 78)
    qualified = df[df["status"] == "QUALIFIED (ГОТОВ К ВХОДУ)"]
    print(f"🎯 Отобрано лидеров в работу: {len(qualified)} из {len(df)}")
    if not qualified.empty:
        top_picks = ", ".join(qualified["coin"].head(config.MAX_ACTIVE_CANDIDATES).tolist())
        print(f"👑 ТОП-КАНДИДАТЫ НА СЕГОДНЯ: [ {top_picks} ]")
    else:
        print("ℹ️ Все активы сейчас слабее BTC или с перегретым фандингом. Защита баланса активна.")

print("=" * 78)
