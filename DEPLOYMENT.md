# Customer Churn Prediction - Deployment Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn joblib scipy fastapi uvicorn pydantic requests
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### 2. Train/Retrain Model
```bash
python scripts/run_pipeline.py
```
This generates:
- `artifacts/cleaned_customer_churn.csv`
- `artifacts/model_results.csv`
- `artifacts/best_model_*.joblib`
- `artifacts/model_report.json`

### 3. Start the API
```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Visit: `http://localhost:8000/docs` for interactive API documentation

### 4. Test the API
```bash
python scripts/test_api.py
```

### 5. Monitor Model Performance
```bash
python scripts/monitor_model.py
```

---

## Project Structure

```
Customer-Churn-Prediction-and-Analysis/
├── artifacts/                          # Model outputs & metadata
│   ├── cleaned_customer_churn.csv
│   ├── model_results.csv
│   ├── best_model_gradient_boosting.joblib
│   └── model_report.json
├── api/                                # FastAPI deployment
│   └── app.py
├── notebooks/                          # Analysis notebooks
│   └── churn_end_to_end.ipynb
├── reports/                            # Documentation
│   ├── final_project_report.md
│   └── presentation_slides.md
├── scripts/                            # Automation
│   ├── run_pipeline.py                # Train pipeline
│   ├── monitor_model.py               # Drift/performance checks
│   └── test_api.py                    # API tests
├── customer_churn_dataset-testing-master.csv
└── README.md
```

---

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35,
    "Gender": "Female",
    "Tenure": 18,
    "Usage_Frequency": 12,
    "Support_Calls": 8,
    "Payment_Delay": 15,
    "Subscription_Type": "Standard",
    "Contract_Length": "Monthly",
    "Total_Spend": 540.0,
    "Last_Interaction": 8,
    "TenureBucket": "1-2y",
    "HighSupport": 1,
    "RecentInteraction": 1,
    "UsagePerTenure": 0.67,
    "SpendPerMonth": 30.0
  }'
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"customers": [...]}'
```

### Model Info
```bash
curl http://localhost:8000/model/info
```

---

## Monitoring Setup

### Automated Drift Detection
Schedule `scripts/monitor_model.py` to run weekly:

**Linux/Mac (cron):**
```bash
0 9 * * 1 /path/to/python /path/to/scripts/monitor_model.py
```

**Windows (Task Scheduler):**
Create a task that runs `python scripts/monitor_model.py` every Monday at 9 AM.

### Alert Triggers
- **Data Drift:** KS-test p-value < 0.05
- **Performance Drop:** F1 score drops >5% absolute
- **Volume Anomaly:** Daily predictions outside 2σ range

### Retraining Strategy
1. Gather ≥5k new labeled observations
2. Run `python scripts/run_pipeline.py`
3. Review `artifacts/model_results.csv`
4. If new model F1 > current + 0.02, deploy
5. Restart API to load new model

---

## Docker Deployment (Optional)

### Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build & Run
```bash
docker build -t churn-api .
docker run -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts churn-api
```

---

## Cloud Deployment

### AWS (Elastic Beanstalk)
```bash
eb init -p python-3.10 churn-api
eb create churn-env
eb deploy
```

### Google Cloud (Cloud Run)
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/churn-api
gcloud run deploy --image gcr.io/PROJECT_ID/churn-api --platform managed
```

### Azure (App Service)
```bash
az webapp up --sku B1 --name churn-api --runtime "PYTHON:3.10"
```

---

## CI/CD Pipeline Example (GitHub Actions)

```yaml
name: Train & Deploy Churn Model

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly
  workflow_dispatch:

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: python scripts/run_pipeline.py
      - run: python scripts/monitor_model.py
      - uses: actions/upload-artifact@v3
        with:
          name: model-artifacts
          path: artifacts/

  deploy:
    needs: train
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v3
      - run: |
          # Deploy to your cloud provider
          # Example: aws s3 cp artifacts/ s3://bucket/
```

---

## Troubleshooting

### Model not loading
- Ensure `artifacts/model_report.json` exists
- Check that `model_path` in JSON points to valid `.joblib` file

### API returns 500 errors
- Check logs: `uvicorn api.app:app --log-level debug`
- Verify all feature names match training schema

### Monitoring script fails
- Ensure production data has same schema as training data
- Check that ground truth labels are available

---

## Support & Contact

- **Technical Issues:** [Your Email]
- **Documentation:** `reports/final_project_report.md`
- **Notebook:** `notebooks/churn_end_to_end.ipynb`
- **Slides:** `reports/presentation_slides.md`

---

## License
[MIT / Your Organization License]

