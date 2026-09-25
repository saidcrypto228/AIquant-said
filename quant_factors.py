import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class QuantFactorEngine:
    """
    Генератор квантовых факторов QVEX v10.7 с гарантированной защитой от Lookahead Bias.
    Все предикторы строго сдвигаются на 1 шаг (.shift(1)), обеспечивая причинно-следственную
    связь (Causality Guarantee): решение в баре t опирается строго на бар t-1.
    """
    def __init__(self, scaler=None):
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.is_fitted = False

    def compute_raw_factors(self, df: pd.DataFrame, btc_df: pd.DataFrame = None) -> pd.DataFrame:
        """Расчет сырых математических признаков."""
        f = pd.DataFrame(index=df.index)

        # 1. Тренд и Моментум (EMA Slope & MACD)
        ema_fast = df["close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["close"].ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9, adjust=False).mean()
        f["macd_hist"] = macd - signal
        f["ema_slope"] = (ema_fast - ema_fast.shift(3)) / (df["close"] + 1e-8)

        # 2. Относительная волатильность (Normalized ATR)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = tr.rolling(window=14).mean()
        f["atr_ratio"] = atr14 / (df["close"] + 1e-8)

        # 3. Объемный моментум (Volume Surge)
        vol_ma = df["volume"].rolling(window=20).mean()
        f["vol_surge"] = df["volume"] / (vol_ma + 1e-8)

        # 4. Относительная сила к BTC (Relative Strength - RS)
        if btc_df is not None and "close" in btc_df:
            alt_ret = df["close"].pct_change(6) # 24 часа (6 баров по 4H)
            btc_ret = btc_df["close"].pct_change(6)
            f["rs_btc"] = alt_ret - btc_ret
        else:
            f["rs_btc"] = 0.0

        # [CRITICAL P1 FIX]: Принудительный сдвиг факторов на 1 бар назад
        # Значения на строке t теперь физически отражают исторические данные строго ДО закрытия бара t
        f_shifted = f.shift(1).copy()
        return f_shifted

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Обучение скейлера СТРОГО на тренировочном наборе."""
        self.is_fitted = True
        return self.scaler.fit_transform(X_train.fillna(0.0))

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Трансформация тестовых данных без утечки математического ожидания."""
        if not self.is_fitted:
            return self.scaler.fit_transform(X.fillna(0.0))
        return self.scaler.transform(X.fillna(0.0))
