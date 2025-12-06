from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

with open("model_xgb.pkl", "rb") as f:
    model_package = pickle.load(f)

model = model_package["model"]
label_encoder = model_package["label_encoder"]
selected_features = model_package["selected_features"]

DISPLAY_LABELS = {
    "PE_Weekly_ATM_LTP": "PE Weekly ATM LTP",
    "PE_Weekly_ATM_PremTO": "PE Weekly ATM Premium Turnover",
    "PE_Weekly_ATM_OI": "PE Weekly ATM Open Interest",
    "CE_Weekly_ATM_LTP": "CE Weekly ATM LTP",
    "CE_Weekly_ATM_PremTO": "CE Weekly ATM Premium Turnover",
    "CE_Weekly_ATM_OI": "CE Weekly ATM Open Interest",
    "PE_Weekly_Pre_1_LTP": "PE Weekly Pre -1 LTP",
    "PE_Weekly_Pre_1_PremTO": "PE Weekly Pre -1 Premium Turnover",
    "PE_Weekly_Pre_1_OI": "PE Weekly Pre -1 Open Interest",
    "PE_Weekly_Pre_2_LTP": "PE Weekly Pre -2 LTP",
    "PE_Weekly_Pre_2_PremTO": "PE Weekly Pre -2 Premium Turnover",
    "PE_Weekly_Pre_2_OI": "PE Weekly Pre -2 Open Interest",
    "CE_Weekly_Post_1_LTP": "CE Weekly Post +1 LTP",
    "CE_Weekly_Post_1_PremTO": "CE Weekly Post +1 Premium Turnover",
    "CE_Weekly_Post_1_OI": "CE Weekly Post +1 Open Interest",
    "CE_Weekly_Post_2_LTP": "CE Weekly Post +2 LTP",
    "CE_Weekly_Post_2_PremTO": "CE Weekly Post +2 Premium Turnover",
    "CE_Weekly_Post_2_OI": "CE Weekly Post +2 Open Interest",
    "Days_to_Expiry": "Days to Expiry",
    "NiftySpot": "Nifty Spot",
}

FEATURE_META = [
    {
        "name": feat,
        "label": DISPLAY_LABELS.get(feat, feat.replace("_", " ")),
        "placeholder": f"Enter {DISPLAY_LABELS.get(feat, feat.replace('_', ' '))}",
    }
    for feat in selected_features
]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    raw_label = None
    error = None
    input_values = {}

    direction_title = None
    direction_desc = None
    trend_type = None
    max_conf = None
    proba_map = {}

    if request.method == "POST":
        try:
            for feat in selected_features:
                value_str = request.form.get(feat)
                if value_str is None or value_str.strip() == "":
                    raise ValueError(f"Missing value for {feat}")
                input_values[feat] = float(value_str)

            input_df = pd.DataFrame([input_values])[selected_features]

            pred_encoded = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]

            pred_label = label_encoder.inverse_transform([pred_encoded])[0]
            raw_label = pred_label

            prediction = f"Model prediction label: {pred_label}"

            for idx, enc_class in enumerate(model.classes_):
                label_val = label_encoder.inverse_transform([enc_class])[0]
                proba_map[str(label_val)] = round(float(proba[idx]) * 100, 2)

            max_conf = round(float(proba[pred_encoded]) * 100, 2)

            trend_type = "neutral"  # default
            label_str = str(pred_label).upper()

            numeric_label = None
            try:
                numeric_label = int(pred_label)
            except Exception:
                pass

            if numeric_label is not None:
                if numeric_label < 0:
                    trend_type = "down"
                    direction_title = "Likely Weekly Downtrend"
                    direction_desc = "The model expects a downward move in Nifty for this weekly expiry window."
                elif numeric_label > 0:
                    trend_type = "up"
                    direction_title = "Likely Weekly Uptrend"
                    direction_desc = "The model expects an upward move in Nifty for this weekly expiry window."
                else:
                    trend_type = "neutral"
                    direction_title = "Range-bound / Neutral Bias"
                    direction_desc = "The model does not see a strong directional edge; price may remain range-bound."
            else:
                if "UP" in label_str:
                    trend_type = "up"
                    direction_title = "Likely Weekly Uptrend"
                    direction_desc = "The model expects bullish price action for this weekly expiry."
                elif "DOWN" in label_str:
                    trend_type = "down"
                    direction_title = "Likely Weekly Downtrend"
                    direction_desc = "The model expects bearish price action for this weekly expiry."
                else:
                    trend_type = "neutral"
                    direction_title = "Range-bound / Neutral Bias"
                    direction_desc = "The model indicates a sideways or low-conviction setup."

        except Exception as e:
            error = f"Error during prediction: {e}"

    return render_template(
        "index.html",
        features=FEATURE_META,
        prediction=prediction,
        raw_label=raw_label,
        error=error,
        input_values=input_values,
        direction_title=direction_title,
        direction_desc=direction_desc,
        trend_type=trend_type,
        max_conf=max_conf,
        proba_map=proba_map,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)