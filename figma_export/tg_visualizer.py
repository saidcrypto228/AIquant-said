#!/usr/bin/env python3
"""
Институциональный визуализатор Telegram для Hyperliquid 4H Swing.
Стандарт: Надежный Telegram MarkdownV2 с полным авто-экранированием.
"""

import os
import io
import re
import urllib.request
import urllib.parse
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np

import bot_config as config

def esc(text: str) -> str:
    """Экранирует абсолютно все служебные символы Telegram MarkdownV2."""
    reserved = r"_*[]()~`>#+-=|{}.!\\"
    return re.sub(r"([%s])" % re.escape(reserved), r"\\\1", str(text))

class TelegramVisualizer:

    @staticmethod
    def send_message(md2_text: str) -> bool:
        token = config.TELEGRAM_BOT_TOKEN
        chat_id = config.TELEGRAM_CHAT_ID
        if not token or not chat_id:
            return False

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": md2_text,
            "parse_mode": "MarkdownV2",
            "disable_web_page_preview": "true"
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers={"User-Agent": "HL-Sentinel"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except urllib.error.HTTPError as e:
            print(f"[-] Telegram sendMessage Error ({e.code}): {e.read().decode('utf-8')}")
            return False
        except Exception as e:
            print(f"[-] Сетевой сбой: {e}")
            return False

    @staticmethod
    def render_trade_chart(df_4h: pd.DataFrame, coin: str, entry_px: float, sl_px: float) -> io.BytesIO:
        plot_df = df_4h.tail(26).copy().reset_index(drop=True)

        plt.style.use("dark_background")
        fig = plt.figure(figsize=(10.5, 6.0), dpi=140)
        fig.patch.set_facecolor("#0b0e14")

        gs = gridspec.GridSpec(2, 1, height_ratios=[3.3, 1], hspace=0.06)
        ax_main = plt.subplot(gs[0])
        ax_vol = plt.subplot(gs[1], sharex=ax_main)

        ax_main.set_facecolor("#11151f")
        ax_vol.set_facecolor("#11151f")

        c_bull = "#0ecb81"
        c_bear = "#f6465d"

        for idx, row in plot_df.iterrows():
            c = c_bull if row["close"] >= row["open"] else c_bear
            body_bottom = min(row["open"], row["close"])
            body_height = max(abs(row["close"] - row["open"]), (row["high"] - row["low"]) * 0.02)

            ax_main.bar(idx, body_height, bottom=body_bottom, color=c, width=0.62, alpha=0.92, zorder=3)
            ax_main.vlines(idx, row["low"], row["high"], color=c, linewidth=1.1, alpha=0.85, zorder=2)
            ax_vol.bar(idx, row["volume"], color=c, width=0.62, alpha=0.65, zorder=3)

        ax_main.plot(plot_df.index, plot_df["ema20_4h"], label="4H EMA-20", color="#f5ac37", linewidth=2.0, zorder=4)
        ax_main.plot(plot_df.index, plot_df["ema50_4h"], label="4H EMA-50", color="#3a86ff", linewidth=1.6, linestyle="--", zorder=4)

        ax_main.fill_between(plot_df.index, plot_df["ema20_4h"], plot_df["ema50_4h"], 
                             where=(plot_df["ema20_4h"] >= plot_df["ema50_4h"]), 
                             color="#f5ac37", alpha=0.08, zorder=1)

        ax_main.axhline(entry_px, color="#00f5d4", linestyle="-.", linewidth=1.3, alpha=0.95, zorder=5)
        ax_main.axhline(sl_px, color="#ff3366", linestyle=":", linewidth=1.5, alpha=0.95, zorder=5)

        x_last = len(plot_df) - 1
        ax_main.annotate(f" ENTRY ${entry_px:,.2f} ", xy=(x_last, entry_px), xytext=(x_last - 5.5, entry_px),
                         fontsize=8, fontweight="bold", color="#0b0e14",
                         bbox=dict(boxstyle="square,pad=0.25", facecolor="#00f5d4", edgecolor="none"),
                         arrowprops=dict(arrowstyle="->", color="#00f5d4", lw=1.2), zorder=6)

        ax_main.annotate(f" SL ${sl_px:,.2f} ", xy=(x_last, sl_px), xytext=(x_last - 5.5, sl_px),
                         fontsize=8, fontweight="bold", color="#ffffff",
                         bbox=dict(boxstyle="square,pad=0.25", facecolor="#ff3366", edgecolor="none"),
                         arrowprops=dict(arrowstyle="->", color="#ff3366", lw=1.2), zorder=6)

        for ax in [ax_main, ax_vol]:
            ax.grid(True, linestyle=":", alpha=0.2, color="#7d8b9f")
            ax.tick_params(colors="#6c757d", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#1f2430")

        plt.setp(ax_main.get_xticklabels(), visible=False)
        time_labels = [dt.strftime("%d.%m %H:%M") for dt in plot_df["datetime"]]
        step = max(1, len(time_labels) // 5)
        ax_vol.set_xticks(range(0, len(time_labels), step))
        ax_vol.set_xticklabels([time_labels[i] for i in range(0, len(time_labels), step)], rotation=0)

        ax_main.set_title(f"HYPERLIQUID L1  |  {coin}/USDC  |  4H PULLBACK", 
                          fontsize=11, fontweight="bold", color="#ffffff", pad=10, loc="left")
        ax_main.legend(loc="upper left", framealpha=0.6, facecolor="#0b0e14", edgecolor="#1f2430", fontsize=8)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf

    @staticmethod
    def send_signal_photo_with_card(df_4h: pd.DataFrame, coin: str, sz: float, entry_px: float, sl_px: float, notional: float, risk_usd: float, rs_pct: float) -> bool:
        token = config.TELEGRAM_BOT_TOKEN
        chat_id = config.TELEGRAM_CHAT_ID
        if not token or not chat_id:
            return False

        try:
            chart_buf = TelegramVisualizer.render_trade_chart(df_4h, coin, entry_px, sl_px)

            # Формируем текст с полным автоматическим экранированием
            t_coin = esc(coin)
            t_entry = esc(f"${entry_px:,.2f}")
            t_sl = esc(f"${sl_px:,.2f}")
            t_sz = esc(f"{sz} {coin}")
            t_notional = esc(f"~${notional:,.1f}")
            t_risk = esc(f"${risk_usd:.2f}")
            t_rs = esc(f"{rs_pct:+.2f}%")

            caption = (
                f"🎯 *СИГНАЛ НА ПОДБОР ЛИКВИДНОСТИ: {t_coin}/USDC*\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🧠 *Физика процесса:*\n"
                f"Рынок в аптренде {esc('(BTC Master Gate активен)')}\\. "
                f"{t_coin} лидирует по силе {esc(f'(RS > +2.0%, факт: {t_rs})')}\\. "
                f"Зафиксирован тест 4H EMA\\-20 с поглощением продаж часовым объемом\\.\n\n"
                f"📋 *Спецификация ордера:*\n"
                f"• *Лимитный Maker\\-вход {esc('(ALO)')}:* `{t_entry}`\n"
                f"• *Нативный Stop\\-Loss {esc('(1.8 ATR)')}:* `{t_sl}`\n"
                f"• *Объем позиции:* `{t_sz}` {esc(f'({t_notional})')}\n"
                f"• *Риск на сделку:* `{t_risk}` {esc('(1.5% equity)')}\n\n"
                f"🛡 *Институциональная защита L1:*\n"
                f"• *Post\\-Only:* 0% комиссии тейкера, исключено проскальзывание\\.\n"
                f"• *Isolated 3x Margin:* изолированный риск на сделку\\.\n"
                f"• *Dead\\-Man Switch:* авто\\-отмена через 15м при сбое сети\\.\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⚡ _{esc('Лимитная заявка выставлена в стакан. Ожидание филла...')}_"
            )

            boundary = "WebKitFormBoundaryMarkdownV2Fixed777"
            body = io.BytesIO()

            def add_field(name, val):
                body.write(f"--{boundary}\r\n".encode("utf-8"))
                body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
                body.write(f"{val}\r\n".encode("utf-8"))

            add_field("chat_id", chat_id)
            add_field("caption", caption)
            add_field("parse_mode", "MarkdownV2")

            body.write(f"--{boundary}\r\n".encode("utf-8"))
            body.write(b'Content-Disposition: form-data; name="photo"; filename="chart.png"\r\n')
            body.write(b"Content-Type: image/png\r\n\r\n")
            body.write(chart_buf.getvalue())
            body.write(b"\r\n")
            body.write(f"--{boundary}--\r\n".encode("utf-8"))

            url = f"https://api.telegram.org/bot{token}/sendPhoto"
            req = urllib.request.Request(
                url,
                data=body.getvalue(),
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "User-Agent": "HL-Sentinel"
                }
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.status == 200
        except urllib.error.HTTPError as e:
            print(f"[-] Telegram MarkdownV2 Photo Error ({e.code}): {e.read().decode('utf-8')}")
            return False
        except Exception as e:
            print(f"[-] Ошибка отправки фото: {e}")
            return False
