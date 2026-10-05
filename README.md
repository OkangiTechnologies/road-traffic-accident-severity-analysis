# Road Traffic Accident Severity — Project

Full write-up: see `REPORT.md`.

## Folder structure

```
RTA_Severity_Project/
├── REPORT.md                  <- full report (steps 1-8, findings, reflection)
├── README.md                  <- this file
├── data/
│   ├── RTA_Dataset.csv        <- raw data (32 cols)
│   ├── rta_clean.csv          <- after step 2 (cleaning)
│   └── rta_readable.csv       <- after step 4 (feature engineering, human-readable)
├── 01_clean_data.py           <- step 2: cleaning
├── 02_eda.py                  <- step 3: exploration / charts
├── 03_feature_engineering.py  <- step 4: new features + encoding
├── 04_train_models.py         <- step 5: train + compare + tune
├── 05_evaluate.py             <- step 6: final test-set evaluation
├── figures/                   <- all charts (target dist., feature breakdowns, confusion matrix)
└── streamlit_app/             <- step 7: the dashboard
    ├── app.py
    ├── requirements.txt
    ├── data/rta_readable.csv
    ├── models/                <- saved model + encoders (no retraining in the app)
    └── figures/07_confusion_matrix.png
```

## Reproduce the pipeline

```bash
pip install -r streamlit_app/requirements.txt
python 01_clean_data.py
python 02_eda.py
python 03_feature_engineering.py
python 04_train_models.py
python 05_evaluate.py
```

## Run the dashboard

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```
