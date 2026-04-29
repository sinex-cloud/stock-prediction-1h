import os
import traceback
import requests
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template, request
from scipy.stats import linregress

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "AAPL_1h.csv")

MODEL_API_URL = os.environ.get("MODEL_API_URL", "http://127.0.0.1:8000/predict")


# ── Helpers ──────────────────────────────────────────────────────────────────
def make_datetime_naive(series_or_index):
    """
    Convert any datetime column/index to timezone-naive datetime.
    Fixes: Invalid comparison between dtype=datetime64[us, UTC] and Timestamp
    """
    dt = pd.to_datetime(series_or_index, errors="coerce", utc=True)
    if isinstance(dt, pd.Series):
        return dt.dt.tz_localize(None)
    return dt.tz_localize(None)


def clean_for_json(value):
    """Convert NaN/inf/numpy values into JSON-safe Python values."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, list):
        return [clean_for_json(v) for v in value]
    if isinstance(value, dict):
        return {k: clean_for_json(v) for k, v in value.items()}
    return value


def safe_round(value, dp=4):
    if value is None:
        return None
    try:
        if pd.isna(value) or np.isinf(value):
            return None
        return round(float(value), dp)
    except Exception:
        return None


# ── Robust CSV loader ────────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV not found at: {CSV_PATH}\n"
            "Edit CSV_PATH in app.py to point to your file."
        )

    with open(CSV_PATH, "r", encoding="utf-8") as fh:
        line0 = fh.readline().rstrip("\n")
        line1 = fh.readline().rstrip("\n")

    first_cell = line0.split(",")[0].strip().lower().strip('"')

    if first_cell in ("price", "ticker", ""):
        df = pd.read_csv(CSV_PATH, header=[0, 1], index_col=0)
        df.columns = df.columns.get_level_values(0)
        df.index = make_datetime_naive(df.index)
        df = df[~df.index.isna()]
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "Datetime"})

    elif first_cell == "datetime":
        df = pd.read_csv(CSV_PATH)
        df = df.rename(columns={df.columns[0]: "Datetime"})
        df["Datetime"] = make_datetime_naive(df["Datetime"])

    else:
        df = pd.read_csv(CSV_PATH, index_col=0)
        df.index = make_datetime_naive(df.index)
        df = df[~df.index.isna()]
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "Datetime"})

    rename = {
        "adj close": "Adj_Close",
        "adjclose": "Adj_Close",
        "adj_close": "Adj_Close",
        "close": "Close",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "volume": "Volume",
        "datetime": "Datetime",
        "date": "Datetime",
    }
    df.columns = [rename.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]

    needed = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing}. Found columns: {list(df.columns)}\n"
            f"File line 1: {line0}\nFile line 2: {line1}"
        )

    df["Datetime"] = make_datetime_naive(df["Datetime"])
    df = df.dropna(subset=["Datetime"])

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Adj_Close" not in df.columns:
        df["Adj_Close"] = df["Close"]
    else:
        df["Adj_Close"] = pd.to_numeric(df["Adj_Close"], errors="coerce").fillna(df["Close"])

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    return df


# ── Feature engineering ──────────────────────────────────────────────────────
def compute_features(df):
    df = df.copy()

    df["target"] = df["Close"].shift(-1)
    df["target_return"] = df["target"] / df["Close"] - 1

    df["return_1h"] = df["Close"].pct_change()
    df["close_lag1"] = df["Close"].shift(1)
    df["close_lag2"] = df["Close"].shift(2)
    df["close_lag3"] = df["Close"].shift(3)

    df["log_return_close"] = np.log(df["Close"] / df["Close"].shift(1))
    df["log_return_open"] = np.log(df["Open"] / df["Open"].shift(1))

    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["oc_change"] = (df["Close"] - df["Open"]) / df["Open"]

    df["ma_5"] = df["Close"].rolling(5).mean()
    df["std_5"] = df["Close"].rolling(5).std()
    df["var_5"] = df["Close"].rolling(5).var()

    k = 2
    df["bb_upper"] = df["ma_5"] + k * df["std_5"]
    df["bb_lower"] = df["ma_5"] - k * df["std_5"]
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]

    n = 5
    df["momentum"] = df["Close"] - df["Close"].shift(n)
    df["rolling_max_5"] = df["Close"].rolling(n).max()
    df["rolling_min_5"] = df["Close"].rolling(n).min()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()

    def slope(series):
        y = series.values
        if np.isnan(y).any():
            return np.nan
        x = np.arange(len(series))
        return linregress(x, y).slope

    df["slope_10"] = df["Close"].rolling(10).apply(slope, raw=False)

    df["prev_close"] = df["Close"].shift(1)
    df["intraday_var"] = df["return_1h"].rolling(8).var()

    daily = df.set_index("Datetime").resample("D").agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
    ).dropna(subset=["Open"])

    daily["Prev_Close"] = daily["Close"].shift(1)
    daily["TR"] = pd.concat(
        [
            daily["High"] - daily["Low"],
            (daily["High"] - daily["Prev_Close"]).abs(),
            (daily["Low"] - daily["Prev_Close"]).abs(),
        ],
        axis=1,
    ).max(axis=1)
    daily["ATR_3"] = daily["TR"].rolling(3).mean()

    return df, daily


# ── Filter helper ────────────────────────────────────────────────────────────
def apply_filters(df):
    df = df.copy()
    df["Datetime"] = make_datetime_naive(df["Datetime"])

    from_arg = request.args.get("from")
    to_arg = request.args.get("to")

    if from_arg:
        from_date = pd.to_datetime(from_arg, errors="coerce")
        if not pd.isna(from_date):
            from_date = from_date.tz_localize(None) if getattr(from_date, "tzinfo", None) else from_date
            df = df[df["Datetime"] >= from_date]

    if to_arg:
        to_date = pd.to_datetime(to_arg, errors="coerce")
        if not pd.isna(to_date):
            to_date = to_date.tz_localize(None) if getattr(to_date, "tzinfo", None) else to_date
            # include the full end day
            to_date = to_date + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            df = df[df["Datetime"] <= to_date]

    return df.reset_index(drop=True)


# Same features as the LSTM notebook training pipeline:
# training_ds = df.drop(columns=["target", "target_return", "return_1h"])
LSTM_FEATURE_COLS = [
    "Adj_Close", "Close", "High", "Low", "Open", "Volume",
    "close_lag1", "close_lag2", "close_lag3",
    "log_return_close", "log_return_open",
    "hl_range", "oc_change",
    "ma_5", "std_5", "var_5",
    "bb_upper", "bb_lower", "bb_width",
    "momentum", "rolling_max_5", "rolling_min_5",
    "RSI_14", "SMA_10", "EMA_10", "slope_10",
    "prev_close",
]

DISPLAY_FEATURE_COLS = [
    "Close", "return_1h", "log_return_close", "log_return_open",
    "close_lag1", "close_lag2", "close_lag3",
    "hl_range", "oc_change",
    "ma_5", "std_5", "var_5", "bb_width",
    "momentum", "rolling_max_5", "rolling_min_5", "RSI_14",
    "SMA_10", "EMA_10", "slope_10", "intraday_var",
]


def call_lstm_api(df):
    df_feat = df.dropna(subset=LSTM_FEATURE_COLS).copy()

    if len(df_feat) < 10:
        return {"error": "Need at least 10 clean rows for LSTM prediction."}, 400

    seq = df_feat[LSTM_FEATURE_COLS].tail(10).astype(float).values.tolist()

    response = requests.post(MODEL_API_URL, json={"sequence": seq}, timeout=30)

    if response.status_code != 200:
        return {
            "error": "Model API returned an error",
            "status_code": response.status_code,
            "details": response.text,
        }, 500

    api_result = response.json()

    # Try to support different API response keys
    prediction_value = (
        api_result.get("prediction")
        or api_result.get("predicted_price")
        or api_result.get("next_pred")
    )

    # If prediction is multi-output: [target_price, target_return]
    if isinstance(prediction_value, list):
        if prediction_value and isinstance(prediction_value[0], list):
            prediction_value = prediction_value[0]
        next_pred = float(prediction_value[0])
        predicted_return_from_model = float(prediction_value[1]) * 100 if len(prediction_value) > 1 else None
    else:
        next_pred = float(prediction_value)

    last_close = float(df_feat["Close"].iloc[-1])
    pred_return = ((next_pred - last_close) / last_close) * 100
    signal = "BUY" if pred_return > 0.15 else "SELL" if pred_return < -0.15 else "HOLD"

    result = {
        "source": "FastAPI LSTM model",
        "next_pred": round(next_pred, 2),
        "last_close": round(last_close, 2),
        "pred_return": round(pred_return, 4),
        "model_predicted_return": safe_round(predicted_return_from_model, 4) if "predicted_return_from_model" in locals() else None,
        "signal": signal,
        "api_raw_response": api_result,
        "sequence_shape": [10, len(LSTM_FEATURE_COLS)],
        "features_used": LSTM_FEATURE_COLS,
    }
    return result, 200


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/debug")
def debug():
    result = {"csv_path": CSV_PATH, "csv_exists": os.path.exists(CSV_PATH)}
    try:
        with open(CSV_PATH, encoding="utf-8") as f:
            result["first_5_lines"] = [f.readline() for _ in range(5)]
        df = load_data()
        df_features, _ = compute_features(df)
        result.update({
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "head": df.head(3).astype(str).to_dict(orient="records"),
            "date_min": str(df["Datetime"].min()),
            "date_max": str(df["Datetime"].max()),
            "null_counts": df.isnull().sum().to_dict(),
            "lstm_features_available": all(c in df_features.columns for c in LSTM_FEATURE_COLS),
            "lstm_feature_count": len(LSTM_FEATURE_COLS),
            "model_api_url": MODEL_API_URL,
            "status": "OK — data loaded successfully",
        })
    except Exception as e:
        result.update({"error": str(e), "traceback": traceback.format_exc(), "status": "ERROR"})
    return jsonify(clean_for_json(result))


@app.route("/api/summary")
def summary():
    try:
        df, _ = compute_features(load_data())
        df = apply_filters(df)
        if len(df) < 2:
            return jsonify({"error": "Not enough rows after filtering"}), 400

        close_df = df.dropna(subset=["Close"])
        last = close_df.iloc[-1]
        prev = close_df.iloc[-2]

        result = {
            "price": safe_round(last["Close"], 2),
            "price_prev": safe_round(prev["Close"], 2),
            "pct_change": safe_round((last["Close"] - prev["Close"]) / prev["Close"] * 100),
            "rsi": safe_round(last["RSI_14"]),
            "return_1h": safe_round(last["return_1h"] * 100),
            "bb_width": safe_round(last["bb_width"]),
            "momentum": safe_round(last["momentum"]),
            "slope_10": safe_round(last["slope_10"]),
            "ma5": safe_round(last["ma_5"], 2),
            "sma10": safe_round(last["SMA_10"], 2),
            "ema10": safe_round(last["EMA_10"], 2),
            "hl_range": safe_round(last["hl_range"] * 100),
            "oc_change": safe_round(last["oc_change"] * 100),
            "total_rows": int(len(df)),
            "date_min": str(df["Datetime"].min().date()),
            "date_max": str(df["Datetime"].max().date()),
        }
        return jsonify(clean_for_json(result))
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/chart/price")
def chart_price():
    try:
        df, _ = compute_features(load_data())
        df = apply_filters(df).dropna(subset=["Close"])
        return jsonify(clean_for_json({
            "labels": df["Datetime"].dt.strftime("%b %d %H:%M").tolist(),
            "close": df["Close"].round(2).tolist(),
            "ma5": df["ma_5"].round(2).tolist(),
            "sma10": df["SMA_10"].round(2).tolist(),
            "ema10": df["EMA_10"].round(2).tolist(),
            "bb_upper": df["bb_upper"].round(2).tolist(),
            "bb_lower": df["bb_lower"].round(2).tolist(),
        }))
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/chart/returns")
def chart_returns():
    try:
        df, _ = compute_features(load_data())
        df = apply_filters(df).dropna(subset=["return_1h"])
        returns = (df["return_1h"] * 100).round(4).tolist()
        return jsonify(clean_for_json({
            "labels": df["Datetime"].dt.strftime("%b %d %H:%M").tolist(),
            "returns": returns,
            "colors": ["#22c55e" if r >= 0 else "#ef4444" for r in returns],
        }))
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/chart/rsi")
def chart_rsi():
    try:
        df, _ = compute_features(load_data())
        df = apply_filters(df).dropna(subset=["RSI_14"])
        return jsonify(clean_for_json({
            "labels": df["Datetime"].dt.strftime("%b %d %H:%M").tolist(),
            "rsi": df["RSI_14"].round(2).tolist(),
        }))
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/chart/volume")
def chart_volume():
    try:
        df, _ = compute_features(load_data())
        df = apply_filters(df).dropna(subset=["Volume"])
        return jsonify(clean_for_json({
            "labels": df["Datetime"].dt.strftime("%b %d %H:%M").tolist(),
            "volume": (df["Volume"] / 1e6).round(2).tolist(),
        }))
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/chart/bollinger")
def chart_bollinger():
    try:
        df, _ = compute_features(load_data())
        df = apply_filters(df).dropna(subset=["bb_upper", "bb_lower"])
        return jsonify(clean_for_json({
            "labels": df["Datetime"].dt.strftime("%b %d %H:%M").tolist(),
            "close": df["Close"].round(2).tolist(),
            "bb_upper": df["bb_upper"].round(2).tolist(),
            "bb_lower": df["bb_lower"].round(2).tolist(),
            "ma5": df["ma_5"].round(2).tolist(),
        }))
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/chart/momentum")
def chart_momentum():
    try:
        df, _ = compute_features(load_data())
        df = apply_filters(df).dropna(subset=["momentum"])
        mom = df["momentum"].round(4).tolist()
        return jsonify(clean_for_json({
            "labels": df["Datetime"].dt.strftime("%b %d %H:%M").tolist(),
            "momentum": mom,
            "colors": ["#22c55e" if m >= 0 else "#ef4444" for m in mom],
            "rolling_max": df["rolling_max_5"].round(2).tolist(),
            "rolling_min": df["rolling_min_5"].round(2).tolist(),
        }))
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/features/table")
def features_table():
    try:
        df, daily = compute_features(load_data())
        df = apply_filters(df)
        valid = df.dropna(subset=["Close", "RSI_14", "ma_5"])
        if valid.empty:
            return jsonify([])

        last = valid.iloc[-1]

        def s(v, dp=4):
            rounded = safe_round(v, dp)
            return "N/A" if rounded is None else str(rounded)

        rows = [
            ("Close", s(last["Close"], 2), "Price"),
            ("Target next hour", s(last["target"], 2), "Price"),
            ("Return 1h", s(last["return_1h"] * 100, 4) + "%", "Return"),
            ("Log Return", s(last["log_return_close"], 6), "Return"),
            ("Close Lag 1", s(last["close_lag1"], 2), "Lag"),
            ("Close Lag 2", s(last["close_lag2"], 2), "Lag"),
            ("Close Lag 3", s(last["close_lag3"], 2), "Lag"),
            ("HL Range", s(last["hl_range"] * 100, 4) + "%", "Candle"),
            ("OC Change", s(last["oc_change"] * 100, 4) + "%", "Candle"),
            ("MA 5", s(last["ma_5"], 2), "Moving Avg"),
            ("SMA 10", s(last["SMA_10"], 2), "Moving Avg"),
            ("EMA 10", s(last["EMA_10"], 2), "Moving Avg"),
            ("Std 5", s(last["std_5"], 4), "Volatility"),
            ("Var 5", s(last["var_5"], 4), "Volatility"),
            ("Intraday Var", s(last["intraday_var"], 6), "Volatility"),
            ("BB Upper", s(last["bb_upper"], 2), "Bollinger"),
            ("BB Lower", s(last["bb_lower"], 2), "Bollinger"),
            ("BB Width", s(last["bb_width"], 6), "Bollinger"),
            ("Momentum 5", s(last["momentum"], 4), "Momentum"),
            ("Roll Max 5", s(last["rolling_max_5"], 2), "Momentum"),
            ("Roll Min 5", s(last["rolling_min_5"], 2), "Momentum"),
            ("RSI 14", s(last["RSI_14"], 2), "Oscillator"),
            ("Slope 10", s(last["slope_10"], 6), "Trend"),
        ]

        atr_clean = daily["ATR_3"].dropna()
        atr_val = atr_clean.iloc[-1] if not atr_clean.empty else None
        rows.append(("ATR 3 daily", s(atr_val, 4), "Volatility"))

        return jsonify([{"feature": r[0], "value": r[1], "category": r[2]} for r in rows])
    except Exception as e:
        # Always return a list here so the frontend rows.map() does not crash
        return jsonify([]), 500


@app.route("/api/stats/missing")
def stats_missing():
    try:
        df, _ = compute_features(load_data())
        cols = [c for c in DISPLAY_FEATURE_COLS + ["target"] if c in df.columns]
        missing = df[cols].isnull().sum()
        return jsonify(clean_for_json({"labels": cols, "counts": missing.tolist()}))
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/prediction")
def prediction():
    """
    Dashboard prediction endpoint.
    IMPORTANT: This does NOT train RandomForest anymore.
    It builds the last 10-row LSTM sequence and calls the FastAPI model at port 8000.
    """
    try:
        df, _ = compute_features(load_data())
        result, status = call_lstm_api(df)
        return jsonify(clean_for_json(result)), status
    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "FastAPI model server is not running. Start it with: uvicorn src.app:app --reload"
        }), 500
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


if __name__ == "__main__":
    print("=" * 60)
    print(f"  CSV  : {CSV_PATH}")
    print(f"  Found: {os.path.exists(CSV_PATH)}")
    print("  Open : http://127.0.0.1:5000")
    print("  Debug: http://127.0.0.1:5000/api/debug")
    print(f"  Model API: {MODEL_API_URL}")
    print("=" * 60)
    app.run(debug=True, port=5000)

