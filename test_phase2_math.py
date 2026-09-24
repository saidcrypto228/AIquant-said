#!/usr/bin/env python3
import time
from hyperliquid.info import Info
from hyperliquid.utils import constants
import bot_config as config
from cs_ranker import CrossSectionalRanker

print("=" * 85)
print("  ФАЗА 2: ВАЛИДАЦИЯ МАТЕМАТИЧЕСКИХ ФАКТОРОВ (OLS RESIDUAL MOMENTUM)")
print("=" * 85)

base_url = constants.TESTNET_API_URL if config.IS_TESTNET else constants.MAINNET_API_URL
info = Info(base_url, skip_ws=True)

ranker = CrossSectionalRanker(info, config.TARGET_COINS)

print(f"[*] Сбор ончейн-данных и расчет регрессий по {len(config.TARGET_COINS)} активам...")
t0 = time.time()
df = ranker.rank_universe(max_funding_apr=45.0)
dt = time.time() - t0

if df.empty:
    print("[-] Ошибка получения данных.")
else:
    print(f"[✓] Квантовый расчет завершен за {dt:.1f} сек!\n")
    print(f"{'РАНГ':<5} | {'АКТИВ':<7} | {'BETA':<6} | {'RS %':<8} | {'Z-RES_MOM':<10} | {'ФАНДИНГ APR':<13} | {'Z-FUND':<7} | {'СТАТУС'}")
    print("-" * 85)
    for _, r in df.iterrows():
        print(f"#{r['rank']:<4} | {r['coin']:<7} | {r['beta_btc']:<6.2f} | {r['rs_to_btc_pct']:+7.2f}% | {r['z_res_mom']:+9.2f}  | {r['funding_apr']:+9.1f}% APR | {r['z_funding']:+6.2f} | {r['status']}")

    print("-" * 85)
    qualified = df[df["status"] == "QUALIFIED (ГОТОВ К ВХОДУ)"]
    print(f"🎯 Отобрано математически чистых лидеров: {len(qualified)} из {len(df)}")
    if not qualified.empty:
        top_picks = ", ".join(qualified["coin"].head(config.MAX_ACTIVE_CANDIDATES).tolist())
        print(f"👑 ТОП-КАНДИДАТЫ ФАЗЫ 2: [ {top_picks} ]")

print("=" * 85)
