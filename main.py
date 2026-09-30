from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.config import FEATURES, TARGET, RISK_CLASSES
from src.preprocess import load_data

def train_and_evaluate():
    root=Path(__file__).resolve().parent; (root/"models").mkdir(exist_ok=True); (root/"reports").mkdir(exist_ok=True)
    data=load_data(); X,y=data[FEATURES],data[TARGET]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
    preprocessor=ColumnTransformer([("numeric",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),FEATURES)])
    model=Pipeline([("preprocessor",preprocessor),("classifier",LogisticRegression(max_iter=2000,random_state=42))])
    model.fit(X_train,y_train); predictions=model.predict(X_test)
    accuracy=accuracy_score(y_test,predictions); precision=precision_score(y_test,predictions,average="weighted",zero_division=0); recall=recall_score(y_test,predictions,average="weighted",zero_division=0); f1=f1_score(y_test,predictions,average="weighted",zero_division=0); matrix=confusion_matrix(y_test,predictions,labels=RISK_CLASSES)
    print("\n===== WATER QUALITY RISK CLASSIFICATION ====="); print(f"Rows          : {len(data)}"); print(f"Training rows : {len(X_train)}"); print(f"Testing rows  : {len(X_test)}"); print(f"Accuracy      : {accuracy:.4f}"); print(f"Precision     : {precision:.4f}"); print(f"Recall        : {recall:.4f}"); print(f"F1-score      : {f1:.4f}"); print("\nClassification Report:"); print(classification_report(y_test,predictions,zero_division=0)); print("Confusion Matrix [Low, Moderate, High]:"); print(matrix)
    model_path=root/"models"/"water_quality_logistic_regression.joblib"; joblib.dump(model,model_path)
    (root/"reports"/"evaluation_report.txt").write_text(f"Water Quality Risk Classification - Evaluation Report\n\nDataset rows: {len(data)}\nTraining rows: {len(X_train)}\nTesting rows: {len(X_test)}\n\nFeatures:\n"+"\n".join(f"- {f}" for f in FEATURES)+f"\n\nModel: Median Imputation + StandardScaler + LogisticRegression\n\nAccuracy: {accuracy:.4f}\nWeighted Precision: {precision:.4f}\nWeighted Recall: {recall:.4f}\nWeighted F1-score: {f1:.4f}\n\nClassification Report:\n"+classification_report(y_test,predictions,zero_division=0)+f"\nConfusion Matrix (rows=true, columns=predicted)\nLabels: Low, Moderate, High\n{matrix}\n",encoding="utf-8")
    print(f"\nModel saved to: {model_path}"); print(f"Report saved to: {root/'reports'/'evaluation_report.txt'}")

if __name__=="__main__": train_and_evaluate()
