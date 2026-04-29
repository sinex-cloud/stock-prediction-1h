import os
import traceback
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template, request
from scipy.stats import linregress
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = Flask(__name__)

# ── CSV path — edit this if your layout is different ─────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "AAPL_1h.csv")


# ── Robust CSV loader ─────────────────────────────────────────────────────────
def load_data():
    """
    Handles every yfinance CSV format:
      Case A — NEW yfinance (>=0.2.38):
        Row 0:  Price, Adj Close, Close, High, Low, Open, Volume
        Row 1:  Ticker, AAPL, AAPL, ...
        Row 2+: data rows  (datetime index)

      Case B — classic yfinance:
        Row 0:  Datetime, Adj Close, Close, High, Low, Open, Volume
        Row 1+: data rows

      Case C — any other date-indexed CSV
    """
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV not found at: {CSV_PATH}\n"
            "Edit CSV_PATH in app.py to point to your file."
        )

    # peek at first two lines to decide format
    with open(CSV_PATH, "r") as fh:
        line0 = fh.readline().rstrip("\n")
        line1 = fh.readline().rstrip("\n")

    first_cell = line0.split(",")[0].strip().lower().strip('"')

    # ── Case A: multi-level header (new yfinance) ─────────────────────────────
    if first_cell in ("price", "ticker", ""):
        df = pd.read_csv(CSV_PATH, header=[0, 1], index_col=0)
        df.columns = df.columns.get_level_values(0)   # keep "Close", "High" etc.
        df.index   = pd.to_datetime(df.index, errors="coerce")
        df         = df[~df.index.isna()]
        df         = df.reset_index()
        # after reset_index the datetime col takes the original index name
        dt_col     = df.columns[0]
        df         = df.rename(columns={dt_col: "Datetime"})

    # ── Case B: "Datetime" as first column ────────────────────────────────────
    elif first_cell == "datetime":
        df = pd.read_csv(CSV_PATH)
        df = df.rename(columns={df.columns[0]: "Datetime"})
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

    # ── Case C: date as index (any other name) ────────────────────────────────
    else:
        df = pd.read_csv(CSV_PATH, index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df       = df[~df.index.isna()]
        df       = df.reset_index()
        df       = df.rename(columns={df.columns[0]: "Datetime"})

    # ── normalise column names ────────────────────────────────────────────────
    RENAME = {
        "adj close": "Adj_Close", "adjclose": "Adj_Close", "adj_close": "Adj_Close",
        "close":  "Close",  "open":   "Open",  "high": "High",
        "low":    "Low",    "volume": "Volume", "datetime": "Datetime", "date": "Datetime",
    }
    df.columns = [RENAME.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]

    # validate
    needed = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns {missing_cols}.\n"
            f"Found columns: {list(df.columns)}\n"
            f"File line 1: {line0}\n"
            f"File line 2: {line1}"
        )

    # clean
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    return df


# ── Feature engineering ───────────────────────────────────────────────────────
def compute_features(df):
    df = df.copy()
    df["target"]           = df["Close"].shift(-1)
    df["return_1h"]        = df["Close"].pct_change()
    df["close_lag1"]       = df["Close"].shift(1)
    df["close_lag2"]       = df["Close"].shift(2)
    df["close_lag3"]       = df["Close"].shift(3)
    df["log_return_close"] = np.log(df["Close"] / df["Close"].shift(1))
    df["log_return_open"]  = np.log(df["Open"]  / df["Open"].shift(1))
    df["hl_range"]         = (df["High"] - df["Low"]) / df["Close"]
    df["oc_change"]        = (df["Close"] - df["Open"]) / df["Open"]

    n = 5
    df["momentum"]      = df["Close"] - df["Close"].shift(n)
    df["rolling_max_5"] = df["Close"].rolling(n).max()
    df["rolling_min_5"] = df["Close"].rolling(n).min()

    delta    = df["Close"].diff()
    gain     = delta.where(delta > 0, 0)
    loss     = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    df["ma_5"]         = df["Close"].rolling(5).mean()
    df["std_5"]        = df["Close"].rolling(5).std()
    df["var_5"]        = df["Close"].rolling(5).var()
    df["intraday_var"] = df["return_1h"].rolling(8).var()

    k = 2
    df["bb_upper"] = df["ma_5"] + k * df["std_5"]
    df["bb_lower"] = df["ma_5"] - k * df["std_5"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["ma_5"].replace(0, np.nan)

    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()

    def slope(series):
        x = np.arange(len(series))
        y = series.values
        if np.isnan(y).any():
            return np.nan
        return linregress(x, y).slope

    df["slope_10"] = df["Close"].rolling(10).apply(slope, raw=False)

    # Daily ATR
    daily = df.set_index("Datetime").resample("D").agg(
        Open=("Open","first"), High=("High","max"),
        Low=("Low","min"),   Close=("Close","last"), Volume=("Volume","sum")
    ).dropna(subset=["Open"])
    daily["Prev_Close"] = daily["Close"].shift(1)
    daily["TR"] = pd.concat([
        daily["High"] - daily["Low"],
        (daily["High"] - daily["Prev_Close"]).abs(),
        (daily["Low"]  - daily["Prev_Close"]).abs()
    ], axis=1).max(axis=1)
    daily["ATR_3"] = daily["TR"].rolling(3).mean()
    return df, daily


# ── Filter helper ─────────────────────────────────────────────────────────────
def apply_filters(df):
    f = request.args.get("from")
    t = request.args.get("to")
    if f:
        df = df[df["Datetime"] >= pd.to_datetime(f)]
    if t:
        df = df[df["Datetime"] <= pd.to_datetime(t)]
    return df.reset_index(drop=True)


# ── ML prediction ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Close","return_1h","log_return_close","log_return_open",
    "close_lag1","close_lag2","close_lag3",
    "hl_range","oc_change",
    "ma_5","std_5","var_5","bb_width",
    "momentum","rolling_max_5","rolling_min_5","RSI_14",
    "SMA_10","EMA_10","slope_10","intraday_var",
]

def run_prediction(df):
    df_ml = df.dropna(subset=FEATURE_COLS + ["target"]).copy()
    if len(df_ml) < 60:
        return None
    X  = df_ml[FEATURE_COLS].values
    y  = df_ml["target"].values
    sp = int(len(X) * 0.8)
    X_tr, X_te = X[:sp], X[sp:]
    y_tr, y_te = y[:sp], y[sp:]
    scaler = StandardScaler()
    model  = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(scaler.fit_transform(X_tr), y_tr)
    preds  = model.predict(scaler.transform(X_te))
    mae    = float(mean_absolute_error(y_te, preds))
    rmse   = float(np.sqrt(mean_squared_error(y_te, preds)))
    dir_acc= float(np.mean(np.sign(np.diff(y_te)) == np.sign(np.diff(preds))) * 100)
    importances = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    last_row    = df.dropna(subset=FEATURE_COLS).iloc[[-1]][FEATURE_COLS].values
    next_pred   = float(model.predict(scaler.transform(last_row))[0])
    last_close  = float(df["Close"].dropna().iloc[-1])
    pred_return = (next_pred - last_close) / last_close * 100
    thr    = 0.15
    signal = "BUY" if pred_return > thr else "SELL" if pred_return < -thr else "HOLD"
    return {
        "mae": round(mae,4), "rmse": round(rmse,4),
        "dir_acc": round(dir_acc,2),
        "next_pred": round(next_pred,2), "last_close": round(last_close,2),
        "pred_return": round(pred_return,4), "signal": signal,
        "importances": {k:round(v*100,2) for k,v in sorted(importances.items(),key=lambda x:-x[1])},
        "test_dates":  df_ml["Datetime"].iloc[sp:].dt.strftime("%b %d %H:%M").tolist(),
        "actual":      [round(v,2) for v in y_te.tolist()],
        "predicted":   [round(v,2) for v in preds.tolist()],
        "train_size":  sp, "test_size": len(X_te),
    }


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── /api/debug — open in browser to see exactly what the loader finds ─────────
@app.route("/api/debug")
def debug():
    result = {"csv_path": CSV_PATH, "csv_exists": os.path.exists(CSV_PATH)}
    try:
        with open(CSV_PATH) as f:
            result["first_5_lines"] = [f.readline() for _ in range(5)]
        df = load_data()
        result.update({
            "shape":      list(df.shape),
            "columns":    list(df.columns),
            "dtypes":     {c: str(t) for c, t in df.dtypes.items()},
            "head":       df.head(3).astype(str).to_dict(orient="records"),
            "date_min":   str(df["Datetime"].min()),
            "date_max":   str(df["Datetime"].max()),
            "null_counts":df.isnull().sum().to_dict(),
            "status":     "OK — data loaded successfully",
        })
    except Exception as e:
        result.update({"error": str(e), "traceback": traceback.format_exc(), "status": "ERROR"})
    return jsonify(result)


@app.route("/api/summary")
def summary():
    try:
        df, _ = compute_features(load_data())
        df = apply_filters(df)
        if len(df) < 2:
            return jsonify({"error": "Not enough rows after filtering"}), 400
        last = df.dropna(subset=["Close"]).iloc[-1]
        prev = df.dropna(subset=["Close"]).iloc[-2]
        def safe(val, dp=4):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            return round(float(val), dp)
        return jsonify({
            "price":      safe(last["Close"],2),
            "price_prev": safe(prev["Close"],2),
            "pct_change": safe((last["Close"]-prev["Close"])/prev["Close"]*100),
            "rsi":        safe(last["RSI_14"]),
            "return_1h":  safe(last["return_1h"]*100),
            "bb_width":   safe(last["bb_width"]),
            "momentum":   safe(last["momentum"]),
            "slope_10":   safe(last["slope_10"]),
            "ma5":        safe(last["ma_5"],2),
            "sma10":      safe(last["SMA_10"],2),
            "ema10":      safe(last["EMA_10"],2),
            "hl_range":   safe(last["hl_range"]*100),
            "oc_change":  safe(last["oc_change"]*100),
            "total_rows": int(len(df)),
            "date_min":   str(df["Datetime"].min().date()),
            "date_max":   str(df["Datetime"].max().date()),
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/chart/price")
def chart_price():
    try:
        df, _ = compute_features(load_data())
        df = apply_filters(df).dropna(subset=["Close"])
        return jsonify({
            "labels":   df["Datetime"].dt.strftime("%b %d %H:%M").tolist(),
            "close":    df["Close"].round(2).tolist(),
            "ma5":      df["ma_5"].round(2).tolist(),
            "sma10":    df["SMA_10"].round(2).tolist(),
            "ema10":    df["EMA_10"].round(2).tolist(),
            "bb_upper": df["bb_upper"].round(2).tolist(),
            "bb_lower": df["bb_lower"].round(2).tolist(),
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/chart/returns")
def chart_returns():
    try:
        df, _ = compute_features(load_data())
        df = apply_filters(df).dropna(subset=["return_1h"])
        returns = (df["return_1h"]*100).round(4).tolist()
        return jsonify({
            "labels":  df["Datetime"].dt.strftime("%b %d %H:%M").tolist(),
            "returns": returns,
            "colors":  ["#22c55e" if r>=0 else "#ef4444" for r in returns],
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/chart/rsi")
def chart_rsi():
    try:
        df, _ = compute_features(load_data())
        df = apply_filters(df).dropna(subset=["RSI_14"])
        return jsonify({
            "labels": df["Datetime"].dt.strftime("%b %d %H:%M").tolist(),
            "rsi":    df["RSI_14"].round(2).tolist(),
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/chart/volume")
def chart_volume():
    try:
        df, _ = compute_features(load_data())
        df = apply_filters(df).dropna(subset=["Volume"])
        return jsonify({
            "labels": df["Datetime"].dt.strftime("%b %d %H:%M").tolist(),
            "volume": (df["Volume"]/1e6).round(2).tolist(),
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/chart/bollinger")
def chart_bollinger():
    try:
        df, _ = compute_features(load_data())
        df = apply_filters(df).dropna(subset=["bb_upper"])
        return jsonify({
            "labels":   df["Datetime"].dt.strftime("%b %d %H:%M").tolist(),
            "close":    df["Close"].round(2).tolist(),
            "bb_upper": df["bb_upper"].round(2).tolist(),
            "bb_lower": df["bb_lower"].round(2).tolist(),
            "ma5":      df["ma_5"].round(2).tolist(),
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/chart/momentum")
def chart_momentum():
    try:
        df, _ = compute_features(load_data())
        df = apply_filters(df).dropna(subset=["momentum"])
        mom = df["momentum"].round(4).tolist()
        return jsonify({
            "labels":      df["Datetime"].dt.strftime("%b %d %H:%M").tolist(),
            "momentum":    mom,
            "colors":      ["#22c55e" if m>=0 else "#ef4444" for m in mom],
            "rolling_max": df["rolling_max_5"].round(2).tolist(),
            "rolling_min": df["rolling_min_5"].round(2).tolist(),
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/features/table")
def features_table():
    try:
        df, daily = compute_features(load_data())
        df = apply_filters(df)
        valid = df.dropna(subset=["Close","RSI_14","ma_5"])
        if valid.empty:
            return jsonify([])
        last = valid.iloc[-1]
        def s(v, dp=4):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "N/A"
            return str(round(float(v), dp))
        rows = [
            ("Close",         s(last["Close"],2),             "Price"),
            ("Target (next)", s(last["target"],2),            "Price"),
            ("Return 1h",     s(last["return_1h"]*100,4)+"%", "Return"),
            ("Log Return",    s(last["log_return_close"],6),  "Return"),
            ("Close Lag 1",   s(last["close_lag1"],2),        "Lag"),
            ("Close Lag 2",   s(last["close_lag2"],2),        "Lag"),
            ("Close Lag 3",   s(last["close_lag3"],2),        "Lag"),
            ("HL Range",      s(last["hl_range"]*100,4)+"%",  "Candle"),
            ("OC Change",     s(last["oc_change"]*100,4)+"%", "Candle"),
            ("MA 5",          s(last["ma_5"],2),              "Moving Avg"),
            ("SMA 10",        s(last["SMA_10"],2),            "Moving Avg"),
            ("EMA 10",        s(last["EMA_10"],2),            "Moving Avg"),
            ("Std 5",         s(last["std_5"],4),             "Volatility"),
            ("Var 5",         s(last["var_5"],4),             "Volatility"),
            ("Intraday Var",  s(last["intraday_var"],6),      "Volatility"),
            ("BB Upper",      s(last["bb_upper"],2),          "Bollinger"),
            ("BB Lower",      s(last["bb_lower"],2),          "Bollinger"),
            ("BB Width",      s(last["bb_width"],6),          "Bollinger"),
            ("Momentum 5",    s(last["momentum"],4),          "Momentum"),
            ("Roll Max 5",    s(last["rolling_max_5"],2),     "Momentum"),
            ("Roll Min 5",    s(last["rolling_min_5"],2),     "Momentum"),
            ("RSI 14",        s(last["RSI_14"],2),            "Oscillator"),
            ("Slope 10",      s(last["slope_10"],6),          "Trend"),
        ]
        atr_val = daily["ATR_3"].dropna().iloc[-1] if not daily["ATR_3"].dropna().empty else None
        rows.append(("ATR 3 (daily)", s(atr_val,4), "Volatility"))
        return jsonify([{"feature":r[0],"value":r[1],"category":r[2]} for r in rows])
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/stats/missing")
def stats_missing():
    try:
        df, _ = compute_features(load_data())
        cols = ["Close","return_1h","log_return_close","close_lag1","close_lag2",
                "close_lag3","hl_range","oc_change","ma_5","std_5","var_5",
                "bb_width","momentum","rolling_max_5","rolling_min_5",
                "RSI_14","SMA_10","EMA_10","slope_10","target"]
        missing = df[cols].isnull().sum()
        return jsonify({"labels": cols, "counts": missing.tolist()})
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/prediction")
def prediction():
    try:
        df, _ = compute_features(load_data())
        result = run_prediction(df)
        if result is None:
            return jsonify({"error": "Need at least 60 clean rows."}), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


if __name__ == "__main__":
    print("=" * 60)
    print(f"  CSV  : {CSV_PATH}")
    print(f"  Found: {os.path.exists(CSV_PATH)}")
    print("  Open : http://127.0.0.1:5000")
    print("  Debug: http://127.0.0.1:5000/api/debug  ← check this first!")
    print("=" * 60)
    app.run(debug=True, port=5000)