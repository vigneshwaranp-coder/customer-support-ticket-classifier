from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

# Create FastAPI application
app = FastAPI(title="Customer Support Ticket Classifier")

# Load tuned models
category_model = joblib.load("tuned_category_model.pkl")
priority_model = joblib.load("tuned_priority_model.pkl")


# Define input format
class TicketRequest(BaseModel):
    text: str = Field(min_length=1)


# Home endpoint
@app.get("/")
def home():
    return {"message": "Customer Support Ticket Classifier API is running!"}


# Health endpoint
@app.get("/health")
def health():
    return {"status": "healthy"}


# Prediction endpoint
@app.post("/predict")
def predict_ticket(ticket: TicketRequest):

    # Get customer ticket text
    ticket_text = [ticket.text]

    # Predict category and priority
    predicted_category = category_model.predict(ticket_text)[0]
    predicted_priority = priority_model.predict(ticket_text)[0]

    # Calculate prediction confidence
    category_confidence = max(category_model.predict_proba(ticket_text)[0])

    priority_confidence = max(priority_model.predict_proba(ticket_text)[0])

    return {
        "category": predicted_category,
        "priority": predicted_priority,
        "category_confidence": round(float(category_confidence), 4),
        "priority_confidence": round(float(priority_confidence), 4),
    }
