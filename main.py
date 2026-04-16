from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import os
import pickle
from typing import Tuple

import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "Heart_Report.pdf")

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)


class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


def build_input_array(data: PatientData) -> np.ndarray:
    return np.array([
        data.age,
        data.sex,
        data.cp,
        data.trestbps,
        data.chol,
        data.fbs,
        data.restecg,
        data.thalach,
        data.exang,
        data.oldpeak,
        data.slope,
        data.ca,
        data.thal,
    ], dtype=float).reshape(1, -1)


def get_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "LOW"
    if probability < 0.7:
        return "MEDIUM"
    return "HIGH"


def build_pdf(data: PatientData, prediction: int, probability: float, risk: str, advice: str, color: colors.Color) -> str:
    content = []
    styles = getSampleStyleSheet()

    content.append(Paragraph("<b><font size=20>Cardiac Shield AI</font></b>", styles["Title"]))
    content.append(Paragraph("<font size=12>Advanced Heart Risk Analysis Report</font>", styles["Normal"]))
    content.append(Spacer(1, 20))

    patient_data = [["Parameter", "Value"]]
    for field, value in data.dict().items():
        patient_data.append([field.upper(), str(value)])

    table = Table(patient_data, hAlign="LEFT")
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ])
    )

    content.append(Paragraph("<b>Patient Clinical Data</b>", styles["Heading2"]))
    content.append(Spacer(1, 10))
    content.append(table)
    content.append(Spacer(1, 20))

    content.append(Paragraph("<b>Risk Assessment</b>", styles["Heading2"]))
    content.append(Spacer(1, 10))
    content.append(Paragraph(f"<b>Prediction:</b> {prediction}", styles["Normal"]))
    content.append(Paragraph(f"<b>Probability:</b> {round(probability * 100, 2)}%", styles["Normal"]))
    content.append(
        Paragraph(
            f"<b>Risk Level:</b> <font color='{color.hexval()}'>{risk}</font>",
            styles["Normal"],
        )
    )
    content.append(Spacer(1, 20))

    content.append(Paragraph("<b>Clinical Interpretation</b>", styles["Heading2"]))
    content.append(
        Paragraph(
            "This result is generated using a machine learning model trained on cardiovascular datasets. "
            "It evaluates key indicators like cholesterol, blood pressure, and heart rate.",
            styles["Normal"],
        )
    )
    content.append(Spacer(1, 20))

    content.append(Paragraph("<b>Medical Recommendation</b>", styles["Heading2"]))
    content.append(Spacer(1, 10))
    content.append(Paragraph(advice, styles["Normal"]))
    content.append(Spacer(1, 20))

    content.append(Paragraph("<b>Lifestyle Guidelines</b>", styles["Heading2"]))
    content.append(
        Paragraph(
            "- Balanced diet with low saturated fat<br/>"
            "- Regular physical activity (30 mins daily)<br/>"
            "- Avoid smoking and alcohol<br/>"
            "- Regular medical checkups",
            styles["Normal"],
        )
    )
    content.append(Spacer(1, 30))

    content.append(
        Paragraph(
            "<i>This report is AI-generated and should not replace professional medical consultation.</i>",
            styles["Normal"],
        )
    )

    doc = SimpleDocTemplate(REPORT_PATH)
    doc.build(content)

    return REPORT_PATH


@app.get("/")
def home() -> dict:
    return {"message": "API Running"}


@app.post("/predict")
def predict(data: PatientData) -> dict:
    input_data = build_input_array(data)
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])
    risk = get_risk_level(probability)

    return {
        "prediction": prediction,
        "probability": round(probability * 100, 2),
        "risk_level": risk,
    }


@app.post("/generate-report")
def generate_report(data: PatientData) -> FileResponse:
    input_data = build_input_array(data)
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])
    risk = get_risk_level(probability)

    if risk == "LOW":
        color = colors.green
        advice = "Maintain healthy lifestyle and routine checkups."
    elif risk == "MEDIUM":
        color = colors.orange
        advice = "Exercise regularly, reduce cholesterol intake, monitor BP."
    else:
        color = colors.red
        advice = "Consult cardiologist immediately. Avoid stress & monitor vitals."

    report_file = build_pdf(data, prediction, probability, risk, advice, color)
    return FileResponse(report_file, media_type="application/pdf", filename="Heart_Report.pdf")