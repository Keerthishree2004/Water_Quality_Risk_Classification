# Water Quality Risk Classification

This final project is aligned with the capstone problem sheet. It uses exactly the five expected inputs: pH, turbidity, temperature, dissolved oxygen and conductivity.

## Included
- 1,200-row synthetic dataset
- Ready-trained Logistic Regression model
- Separate source files
- EDA outputs
- Evaluation report
- Streamlit application
- Batch prediction sample CSV
- Complete project guide

## Start immediately after extraction
Open the extracted folder in VS Code, open Terminal, then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dataset and trained model are already included, so you do not need to train the model just to open the application.

## Optional
```bash
python main.py
python -m src.eda
python predict.py
python data/generate_dataset.py
```

## Folder structure
```text
Water_Quality_Risk_Classification_FINAL/
├── app.py
├── main.py
├── predict.py
├── requirements.txt
├── README.md
├── START_APP.bat
├── TRAIN_MODEL.bat
├── data/
│   ├── water_quality_dataset.csv
│   ├── sample_batch_input.csv
│   └── generate_dataset.py
├── models/
│   └── water_quality_logistic_regression.joblib
├── reports/
│   ├── evaluation_report.txt
│   ├── sample_results.txt
│   ├── class_distribution.png
│   └── feature_distributions.png
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocess.py
│   └── eda.py
└── docs/
    └── PROJECT_GUIDE.docx
```

## Note
The dataset is synthetic for academic reproducibility and is not certified laboratory data.
