# Customer Support Ticket Classifier

## Project Overview

This project uses Machine Learning and Deep Learning to classify customer support tickets automatically.

It predicts:
- Ticket Category
- Ticket Priority

## Dataset

The original synthetic dataset contains 1,000 customer support tickets.

After text cleaning and duplicate removal, 500 unique tickets remained.

Categories:
- Account
- Billing
- Order
- Technical

Priorities:
- High
- Low
- Medium

The cleaned dataset was split into:
- Training: 350 tickets
- Validation: 75 tickets
- Testing: 75 tickets

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- DistilBERT
- PyTorch
- Hugging Face Transformers
- FastAPI
- Uvicorn
- Pytest

## Models Implemented

### 1. TF-IDF + Logistic Regression

Used as the baseline Machine Learning model.

Hyperparameter tuning was performed using GridSearchCV.

Separate models were trained for category and priority classification.

### 2. DistilBERT

Fine-tuned for category and priority classification using:
- PyTorch
- Hugging Face Transformers
- Hugging Face Trainer API

Separate models were trained for category and priority classification.

## Model Evaluation

Final test results using the cleaned dataset:

| Model | Category Test Accuracy | Category Macro F1 | Priority Test Accuracy | Priority Macro F1 |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 100% | 1.0000 | 76% | 0.6859 |
| DistilBERT Trainer | 100% | 1.0000 | 46.67% | 0.2121 |

### Training and Inference Time Comparison

| Model | Training Time | Inference Time per Ticket |
|---|---|---:|
| TF-IDF + Logistic Regression | Not recorded | 12.95 ms |
| DistilBERT Trainer | 326.81 seconds | 381.97 ms |

**Notes:**
- DistilBERT training time includes category and priority model training.
- Inference time measures both category and priority predictions together.
- Baseline training time was not recorded during the original training.
- Inference timings are measured on a CPU and may vary between runs.

### Model Selection and Justification

1. Both models achieved 100% category test accuracy.
2. TF-IDF + Logistic Regression performed better in priority classification, achieving 76% accuracy compared to DistilBERT's 46.67%.
3. TF-IDF is faster, taking 12.95 ms per ticket compared to DistilBERT's 381.97 ms.
4. TF-IDF requires fewer computational resources and is simpler to deploy.
5. Therefore, TF-IDF + Logistic Regression was selected for the FastAPI application because it provides better priority classification performance and faster inference.

**Note:** The dataset is synthetic and contains generated text patterns. These results are illustrative and should not be interpreted as real-world performance.

## FastAPI Application

The project provides three endpoints:

- `GET /` — Checks whether the API is running.
- `GET /health` — Checks the API health status.
- `POST /predict` — Predicts ticket category and priority and returns confidence scores.

The API loads the trained models when the application starts.

### Example Request

```json
{
  "text": "My credit card was charged twice for the same order. Please help me get a refund."
}
```

### Example Response

```json
{
  "category": "Billing",
  "priority": "High",
  "category_confidence": 0.3945,
  "priority_confidence": 0.9769
}
```

The confidence values are model probabilities and are not guarantees of prediction correctness.

### Example curl Commands

**Health Check**

```bash
curl http://127.0.0.1:8000/health
```

**Predict Ticket**

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"text\":\"My credit card was charged twice for the same order. Please help me get a refund.\"}"
```

## API Validation

The API validates ticket input and rejects empty or missing text.

An empty ticket returns HTTP 422 (Unprocessable Entity).

## Automated Testing

Automated tests were implemented using Pytest and FastAPI TestClient.

Five tests passed:

1. Home endpoint test
2. Health endpoint test
3. Ticket prediction test
4. Empty ticket validation test
5. Missing text validation test

Run the tests using:

```bash
python -m pytest test_app.py -v
```

## How to Run the Project

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn joblib matplotlib torch transformers accelerate fastapi uvicorn httpx pytest
```

### 2. Start the API

```bash
python -m uvicorn app:app --reload
```

### 3. Open API Documentation

http://127.0.0.1:8000/docs

### 4. Run Automated Tests

```bash
python -m pytest test_app.py -v
```

## Project Files
- `README.md` — Project overview, setup instructions, model evaluation and API documentation.

- `ticket_classifier_part1.ipynb` — Data cleaning, model development, tuning and evaluation.
- `app.py` — FastAPI application.
- `test_app.py` — Automated API tests.
- `synthetic_customer_support_tickets_1000.csv` — Original synthetic dataset.
- `cleaned_customer_support_tickets.csv` — Cleaned dataset containing 500 unique tickets.
- `tuned_category_model.pkl` — Saved tuned category model.
- `tuned_priority_model.pkl` — Saved tuned priority model.
- `ticket_tfidf.pkl` — Original TF-IDF vectorizer.
- `ticket_category_model.pkl` — Original category model.
- `ticket_priority_model.pkl` — Original priority model.
- `final_category_model/` — Saved category DistilBERT model and label encoder.
- `final_priority_model/` — Saved priority DistilBERT model and label encoder.
- `final_model_comparison.csv` — Final model comparison.
- `tuned_baseline_results.csv` — Tuned baseline evaluation results.
- `distilbert_test_metrics.pkl` — DistilBERT test metrics.
- `priority_error_analysis.csv` — Priority prediction error analysis.