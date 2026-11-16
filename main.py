
import joblib
import re
from pathlib import Path

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Project folder location
BASE_DIR = Path(__file__).resolve().parent

# Load saved ML model and TF-IDF vectorizer
model = joblib.load(BASE_DIR / "ticket_model.pkl")
tfidf = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")

# Create FastAPI application
app = FastAPI(
    title="Customer Support Ticket Classification",
    description="Predict customer support ticket categories using Machine Learning."
)

# Clean input text
def clean_text(text):
    text = str(text).lower().strip()
    return text

# Home page
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Customer Support Ticket Classifier</title>
        <style>
            body {
                font-family: Arial;
                background: #eef2f7;
                padding: 40px;
            }
            .container {
                background: white;
                max-width: 650px;
                margin: auto;
                padding: 30px;
                border-radius: 12px;
            }
            textarea {
                width: 100%;
                height: 120px;
                padding: 10px;
            }
            button {
                background: #2563eb;
                color: white;
                padding: 12px 20px;
                border: none;
                cursor: pointer;
                margin-top: 12px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Customer Support Ticket Classifier</h1>
            <p>Enter your ticket description below.</p>
            <form action="/predict" method="post">
                <textarea name="ticket_description"
                placeholder="Enter your problem here..." required></textarea>
                <br>
                <button type="submit">Predict Category</button>
            </form>
            <p><a href="/docs">API Documentation</a></p>
        </div>
    </body>
    </html>
    """

# Web form prediction
@app.post("/predict", response_class=HTMLResponse)
def predict(ticket_description: str = Form(...)):
    cleaned_text = clean_text(ticket_description)

    features = tfidf.transform([cleaned_text])
    prediction = model.predict(features)[0]

    return f"""
    <html>
    <body style="font-family:Arial; padding:40px;">
        <h2>Prediction Result</h2>
        <p><b>Ticket:</b> {ticket_description}</p>
        <h3>Predicted Category: {prediction}</h3>
        <a href="/">Try Another Ticket</a>
    </body>
    </html>
    """

# JSON API request format
class TicketRequest(BaseModel):
    ticket_description: str

# JSON API prediction
@app.post("/api/predict")
def api_predict(request: TicketRequest):
    cleaned_text = clean_text(request.ticket_description)

    features = tfidf.transform([cleaned_text])
    prediction = model.predict(features)[0]

    return {
        "ticket_description": request.ticket_description,
        "predicted_category": prediction
    }

# Health check
@app.get("/health")
def health():
    return {"status": "running"}
