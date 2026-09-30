from pathlib import Path
import joblib
import pandas as pd
from src.config import RISK_CLASSES
MODEL_PATH=Path(__file__).resolve().parent/"models"/"water_quality_logistic_regression.joblib"

def predict_sample(values):
    model=joblib.load(MODEL_PATH); row=pd.DataFrame([values]); prediction=model.predict(row)[0]; probabilities=model.predict_proba(row)[0]; return prediction,dict(zip(model.classes_,probabilities))

def main():
    if not MODEL_PATH.exists(): print("Model not found. Run: python main.py"); return
    values={"ph":float(input("pH: ")),"turbidity_ntu":float(input("Turbidity (NTU): ")),"temperature_c":float(input("Temperature (°C): ")),"dissolved_oxygen_mg_l":float(input("Dissolved Oxygen (mg/L): ")),"conductivity_us_cm":float(input("Conductivity (µS/cm): "))}
    prediction,probabilities=predict_sample(values); print("\n"+"="*45); print(f"Predicted Risk Category: {prediction}"); print("="*45)
    for label in RISK_CLASSES: print(f"{label:10s}: {probabilities.get(label,0.0)*100:.2f}%")

if __name__=="__main__": main()
