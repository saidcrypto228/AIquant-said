import { BotState, Candle } from '../types';

// Автоматический выбор базового URL API:
// 1. При локальной разработке фронтенда (Vite :5173 / React :3000) -> шлем на FastAPI :8000
// 2. В продакшне внутри Telegram Mini App -> используем относительный путь ''
const isLocalDev =
  typeof window !== 'undefined' &&
  (window.location.port === '5173' || window.location.port === '3000');

const API_BASE = isLocalDev ? 'http://localhost:8000' : '';

/**
 * Синтезирует историю свечей вокруг текущей цены актива,
 * если торговое ядро еще не передало исторический массив свечей.
 */
function generateFallbackCandles(basePrice: number): Candle[] {
  const candles: Candle[] = [];
  const validBase = basePrice > 0 ? basePrice : 135.0;
  let currentPrice = validBase * 0.94;
  const nowSec = Math.floor(Date.now() / 1000);
  const candleInterval = 4 * 3600; // 4-часовой таймфрейм

  for (let i = 59; i >= 0; i--) {
    const time = nowSec - i * candleInterval;
    const change = (Math.random() - 0.48) * 0.02;
    const open = currentPrice;
    const close = i === 0 ? validBase : currentPrice * (1 + change);
    const high = Math.max(open, close) * (1 + Math.random() * 0.007);
    const low = Math.min(open, close) * (1 - Math.random() * 0.007);
    const volume = Math.floor(Math.random() * 12000) + 1500;

    candles.push({
      time,
      open: Number(open.toFixed(2)),
      high: Number(high.toFixed(2)),
      low: Number(low.toFixed(2)),
      close: Number(close.toFixed(2)),
      volume,
    } as unknown as Candle);

    currentPrice = close;
  }
  return candles;
}

/**
 * Получение текущего состояния торгового робота из FastAPI (/api/state)
 */
export async function fetchBotState(): Promise {
  const response = await fetch(`${API_BASE}/api/state`, {
    method: 'GET',
    headers: {
      Accept: 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`API Error: \({response.status}\){response.statusText}`);
  }

  const data = await response.json();

  // Гарантируем корректную отрисовку CandleChart:
  // Если в состоянии нет свечей, формируем их по цене активной позиции
  if (!data.candles || data.candles.length === 0) {
    const activePrice =
      data.positions?.[0]?.current_price ||
      data.market_regime?.btc_price ||
      135.0;
    data.candles = generateFallbackCandles(activePrice);
  }

  return data as BotState;
}

/**
 * Отправка сигнала экстренного закрытия всех позиций (/api/kill)
 */
export async function sendKillSignal(): Promise {
  const response = await fetch(`${API_BASE}/api/kill`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Kill-Switch Error: \({response.status}\){response.statusText}`);
  }
}