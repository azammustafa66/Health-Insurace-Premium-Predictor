# Health Insurance Cost Predictor

This project is an **end-to-end machine learning solution** designed to predict **health insurance premiums** based on user demographics, lifestyle, and medical history. It demonstrates the complete ML workflow — from **data segmentation and model training** to **real-time prediction via a web app**.

---

## Project Overview

- **Goal:** Predict health insurance premiums accurately and efficiently for different age groups.
- **Approach:** Split the dataset into two cohorts — **young (≤ 25 years)** and **rest (> 25 years)** — and train specialized models for each group.
- **Deployment:** Implemented an interactive **Streamlit app** that allows users to input their details and instantly receive a predicted premium.

---

## How It Works

1. **Data Segmentation:**

   - `data_segmentation.ipynb` divides the original dataset into two groups: `young_premiums.xlsx` and `rest_premiums.xlsx`.
2. **Model Training:**

   - `ml_premium_prediction_young.ipynb` trains the **young model**.
   - `ml_premium_prediction_rest.ipynb` trains the **rest model**.
   - Both notebooks output models and scalers saved as `.joblib` files in the `artifacts/` folder.
3. **Prediction Logic:**

   - The `prediction_helper.py` script handles input preprocessing, categorical encoding, risk scoring, scaling, and model inference.
   - Depending on the user’s age, it automatically selects the correct model (`model_young` or `model_rest`).
4. **Web App:**

   - `app.py` provides a simple, interactive **Streamlit** interface where users can input features (age, region, BMI, smoking status, etc.) and view predicted premiums.

---

## Key Features

- **Age-based modeling:** Two tailored models for young and adult users.
- **Medical risk scoring:** Calculates a normalized score from conditions like diabetes, heart disease, and thyroid.
- **Scalable architecture:** Models and scalers are modular and easily replaceable.
- **Clean UI:** Built with Streamlit for fast, user-friendly predictions.

---

## Tech Stack

**Languages & Tools:** Python, Pandas, NumPy, Scikit-learn, XGBoost, Streamlit
**Visualization:** Matplotlib, Seaborn
**Model Persistence:** Joblib

---

## Project Structure

```
project/
│
├── artifacts/                     # Trained models and scalers
│   ├── model_young.joblib
│   ├── model_rest.joblib
│   ├── scaler_young.joblib
│   └── scaler_rest.joblib
│
├── dataset/                       # Datasets used for training
│   ├── premiums.xlsx
│   ├── young_premiums.xlsx
│   └── rest_premiums.xlsx
│   
│── data_segmentation.ipynb
│── ml_premium_prediction_young.ipynb
│── ml_premium_prediction_rest.ipynb
│── ml_premium_prediction.ipynb
|
├── helper.py           # Preprocessing and prediction pipeline
├── app.py                         # Streamlit web interface
├── requirements.txt               # Dependencies
└── README.md
```

---

## Getting Started

**1. Install dependencies:**

```bash
pip install -r requirements.txt
```

**2. Run the app:**

```bash
streamlit run app.py
```

**3. Open in your browser:**

```
http://localhost:8501
```

---

## Example Use Case

> A user enters their details — age, region, BMI category, smoking status, income, and medical history — into the app.
> The backend selects the correct model, processes the data, and predicts their **estimated annual insurance premium** in Indian Rupees (₹).

---

## Why This Project Matters

This project demonstrates:

- Building a **real-world ML system** from raw data to deployment.
- **Modular architecture** for maintainable, scalable machine learning.
- Strong integration of **data science, feature engineering, and web development**.

It’s ideal for showcasing practical experience in **applied machine learning**, **data-driven modeling**, and **end-user deployment**.
