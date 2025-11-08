from pathlib import Path
import joblib
import pandas as pd
from typing import Dict, Any

ARTIFACTS_DIR = Path("artifacts")


# Attempt to load artifacts at import time (fails loudly so the app can show an error)
def _load_artifact(name: str):
    path = ARTIFACTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return joblib.load(path)


try:
    model_young = _load_artifact("model_young.joblib")
    model_rest = _load_artifact("model_rest.joblib")
except Exception:
    # Defer raising until predict is called so the Streamlit UI can still show.
    model_young = None
    model_rest = None

# Scalers might be stored either as raw scaler objects or as dicts with metadata
try:
    scaler_young = _load_artifact("scaler_young.joblib")
except Exception:
    scaler_young = None

try:
    scaler_rest = _load_artifact("scaler_rest.joblib")
except Exception:
    scaler_rest = None

# Risk mapping (lowercased keys for robustness)
_RISK_SCORES = {
    "diabetes": 6,
    "heart disease": 8,
    "high blood pressure": 6,
    "thyroid": 5,
    "no disease": 0,
    "none": 0,
}


def calculate_normalized_risk(medical_history: str) -> float:
    """Return normalized risk score in [0, 1] based on medical_history string.

    Accepts compound histories separated by ' & ' or ',' and is case-insensitive.
    Unknown conditions default to 0.
    """
    if not isinstance(medical_history, str) or medical_history.strip() == "":
        return 0.0

    # Normalize separators and split
    parts = [
        p.strip().lower() for p in medical_history.replace(",", " & ").split(" & ")
    ]
    # Sum risk scores
    total = sum(_RISK_SCORES.get(p, 0) for p in parts)

    # max possible: heart disease (8) + next highest (6) = 14 — keep consistent with the original helper
    max_score = 14
    normalized = float(total) / max_score if max_score > 0 else 0.0
    return min(max(normalized, 0.0), 1.0)


# Define expected columns (same order used during model training)
_EXPECTED_COLUMNS = [
    "age",
    "number_of_dependants",
    "income_lakhs",
    "insurance_plan",
    "genetical_risk",
    "normalized_risk_score",
    "gender_Male",
    "region_Northwest",
    "region_Southeast",
    "region_Southwest",
    "marital_status_Unmarried",
    "bmi_category_Obesity",
    "bmi_category_Overweight",
    "bmi_category_Underweight",
    "smoking_status_Occasional",
    "smoking_status_Regular",
    "employment_status_Salaried",
    "employment_status_Self-Employed",
]

_INSURANCE_ENCODING = {"Bronze": 1, "Silver": 2, "Gold": 3}


def _ensure_scaler_format(scaler_obj: Any) -> Dict[str, Any]:
    """Return a dict with keys 'scaler' and 'cols_to_scale'. If scaler_obj is already
    a dict with these keys, validate and return it; if it is a scaler, assume it was
    fit on ['age', 'income_lakhs'] by default.
    """
    if scaler_obj is None:
        return None
    if isinstance(scaler_obj, dict):
        if "scaler" in scaler_obj and "cols_to_scale" in scaler_obj:
            return scaler_obj
        else:
            raise ValueError(
                'Scaler dict must contain keys "scaler" and "cols_to_scale"'
            )
    else:
        # assume scaler works on ['age', 'income_lakhs']
        return {"scaler": scaler_obj, "cols_to_scale": ["age", "income_lakhs"]}


def preprocess_input(input_dict: Dict[str, Any]) -> pd.DataFrame:
    """Convert raw input dict into a model-ready DataFrame following expected columns.

    This function:
    - fills the expected columns with defaults of 0
    - maps categorical choices to one-hot / numeric encodings
    - computes normalized_risk_score
    - applies appropriate scaler (young vs rest) if present
    """
    # Basic validation
    if not isinstance(input_dict, dict):
        raise ValueError("input_dict must be a dict")

    df = pd.DataFrame(0, index=[0], columns=_EXPECTED_COLUMNS)

    # Numeric fields with defaults
    df.at[0, "age"] = int(input_dict.get("Age", 0))
    df.at[0, "number_of_dependants"] = int(input_dict.get("Number of Dependants", 0))
    df.at[0, "income_lakhs"] = float(input_dict.get("Income in Lakhs", 0.0))
    df.at[0, "genetical_risk"] = float(input_dict.get("Genetical Risk", 0.0))

    # Insurance
    df.at[0, "insurance_plan"] = _INSURANCE_ENCODING.get(
        input_dict.get("Insurance Plan", "Bronze"), 1
    )

    # Categorical -> one-hot (only set columns that are present in _EXPECTED_COLUMNS)
    if input_dict.get("Gender", "").lower() == "male":
        df.at[0, "gender_Male"] = 1

    region = str(input_dict.get("Region", "")).strip()
    if region == "Northwest":
        df.at[0, "region_Northwest"] = 1
    elif region == "Southeast":
        df.at[0, "region_Southeast"] = 1
    elif region == "Southwest":
        df.at[0, "region_Southwest"] = 1

    if str(input_dict.get("Marital Status", "")).strip() == "Unmarried":
        df.at[0, "marital_status_Unmarried"] = 1

    bmi = str(input_dict.get("BMI Category", "")).strip()
    if bmi == "Obesity":
        df.at[0, "bmi_category_Obesity"] = 1
    elif bmi == "Overweight":
        df.at[0, "bmi_category_Overweight"] = 1
    elif bmi == "Underweight":
        df.at[0, "bmi_category_Underweight"] = 1

    smoking = str(input_dict.get("Smoking Status", "")).strip()
    if smoking == "Occasional":
        df.at[0, "smoking_status_Occasional"] = 1
    elif smoking == "Regular":
        df.at[0, "smoking_status_Regular"] = 1

    employment = str(input_dict.get("Employment Status", "")).strip()
    if employment == "Salaried":
        df.at[0, "employment_status_Salaried"] = 1
    elif employment == "Self-Employed":
        df.at[0, "employment_status_Self-Employed"] = 1

    # normalized risk score
    df.at[0, "normalized_risk_score"] = calculate_normalized_risk(
        str(input_dict.get("Medical History", ""))
    )

    # Apply scaling if scaler objects are available
    age = int(df.at[0, "age"])
    if age <= 25:
        scaler_obj = _ensure_scaler_format(scaler_young)
    else:
        scaler_obj = _ensure_scaler_format(scaler_rest)

    if scaler_obj is not None:
        cols = scaler_obj["cols_to_scale"]
        scaler = scaler_obj["scaler"]
        # Ensure columns exist in df
        missing = [c for c in cols if c not in df.columns]
        if missing:
            # If scaler expects columns not present, raise a helpful error
            raise ValueError(
                f"Scaler expects columns that are not in the input DataFrame: {missing}"
            )
        # transform
        df[cols] = scaler.transform(df[cols])

    return df


def predict(input_dict: Dict[str, Any]) -> int:
    """Run preprocessing and model prediction and return the predicted premium as int.

    Uses the `young` model for age <= 25 and `rest` model otherwise.
    """
    if model_young is None or model_rest is None:
        raise RuntimeError(
            "Models not loaded. Ensure artifacts/model_young.joblib and model_rest.joblib exist in the artifacts folder."
        )

    df = preprocess_input(input_dict)
    if int(df.at[0, "age"]) <= 25:
        pred = model_young.predict(df)
    else:
        pred = model_rest.predict(df)

    # Return integer rupee value (or currency unit). Clip to non-negative.
    val = float(pred[0])
    return int(max(0, round(val)))


# Optional: small test when running module directly
if __name__ == "__main__":
    sample = {
        "Age": 30,
        "Number of Dependants": 0,
        "Income in Lakhs": 12,
        "Genetical Risk": 1,
        "Insurance Plan": "Silver",
        "Employment Status": "Salaried",
        "Gender": "Male",
        "Marital Status": "Unmarried",
        "BMI Category": "Normal",
        "Smoking Status": "No Smoking",
        "Region": "Northwest",
        "Medical History": "No Disease",
    }
    try:
        print("Sample prediction:", predict(sample))
    except Exception as e:
        print("Error when testing prediction helper:", e)
