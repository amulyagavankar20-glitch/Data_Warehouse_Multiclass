"""Pydantic schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    """Input schema for disease prediction and anomaly scoring."""

    age: float = Field(..., ge=18, le=100, description="Patient age")
    age_normalized: float = Field(..., description="Age normalized value")
    bmi: float = Field(..., ge=10, le=60, description="Body Mass Index")
    HbA1c_level: float = Field(..., ge=3, le=15, description="HbA1c blood sugar level")
    glucose: float = Field(..., ge=20, le=600, description="Blood glucose level")
    cholesterol: float = Field(..., ge=50, le=600, description="Total cholesterol")
    sleep_hours: float = Field(..., ge=0, le=24, description="Average sleep hours")
    triglycerides: float = Field(..., ge=0, le=2000, description="Triglyceride level")
    crp_level: float = Field(..., ge=0, le=200, description="CRP inflammation marker")
    homocysteine_level: float = Field(..., ge=0, le=100, description="Homocysteine level")
    systolic_bp: float = Field(..., ge=50, le=250, description="Systolic blood pressure")
    diastolic_bp: float = Field(..., ge=30, le=150, description="Diastolic blood pressure")
    blood_pressure: float = Field(..., ge=30, le=250, description="Blood pressure reading")
    alcohol_intake: float = Field(..., ge=0, description="Alcohol intake frequency")
    salt_intake: float = Field(..., ge=0, description="Salt intake level")
    heart_rate: float = Field(..., ge=30, le=220, description="Resting heart rate")
    hdl: float = Field(..., ge=10, le=100, description="HDL cholesterol")
    ldl: float = Field(..., ge=10, le=300, description="LDL cholesterol")

    age_level: str = Field(..., description="Age level category")
    gender: str = Field(..., description="Gender")
    bmi_level: str = Field(..., description="BMI category")
    smoking: str = Field(..., description="Smoking status")
    physical_activity: str = Field(..., description="Physical activity level")
    family_history: str = Field(..., description="Family history of disease")
    stress_level: str = Field(..., description="Stress level category")
    low_hdl_cholesterol: str = Field(..., description="Low HDL cholesterol indicator")
    high_ldl_cholesterol: str = Field(..., description="High LDL cholesterol indicator")
    high_blood_pressure: str = Field(..., description="High blood pressure indicator")
    sugar_consumption: str = Field(..., description="Sugar consumption level")
    education_level: str = Field(..., description="Education level")
    employment_status: str = Field(..., description="Employment status")


class PredictionResponse(BaseModel):
    """Response schema for disease prediction."""

    predicted_sublabel: str = Field(..., description="Predicted disease sublabel")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    probabilities: dict[str, float] = Field(
        ..., description="Probability for each class"
    )


class AnomalyScoreResponse(BaseModel):
    """Response schema for anomaly scoring."""

    iso_score: float = Field(..., description="Isolation Forest score (lower = more anomalous)")
    ae_mse: float = Field(..., description="Autoencoder reconstruction MSE (higher = more anomalous)")
    is_anomaly_isolation: bool = Field(..., description="Anomaly flag from Isolation Forest")
    is_anomaly_svm: bool = Field(..., description="Anomaly flag from One-Class SVM")


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str
    model_loaded: bool
    model_name: str
