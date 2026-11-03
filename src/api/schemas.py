"""Pydantic request / response schemas for the serving API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]


class PatientFeatures(BaseModel):
    """One raw clinical record. Bounds are physiological, not cohort, limits."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "age_level": "Mature",
                "gender": "Female",
                "bmi_level": "Overweight",
                "smoking": "Never",
                "age": 52,
                "bmi": 27.4,
                "HbA1c_level": 6.1,
                "glucose": 140,
                "cholesterol": 215,
                "sleep_hours": 6.8,
                "triglycerides": 240,
                "physical_activity": "Moderate",
                "family_history": "Yes",
                "stress_level": "Medium",
                "low_hdl_cholesterol": "No",
                "high_ldl_cholesterol": "Yes",
                "blood_pressure": 150,
                "high_blood_pressure": "Yes",
                "sugar_consumption": "Medium",
                "crp_level": 6.2,
                "homocysteine_level": 12.5,
                "systolic_bp": 138,
                "diastolic_bp": 88,
                "alcohol_intake": 10,
                "salt_intake": 8.5,
                "heart_rate": 76,
                "hdl": 55,
                "ldl": 130,
                "education_level": "Secondary",
                "employment_status": "Employed",
            }
        },
    )

    age_level: Literal["Young", "Middle", "Mature", "Senior"]
    gender: Literal["Female", "Male"]
    bmi_level: Literal["Underweight", "Normal", "Overweight", "Obese"]
    smoking: Literal["Current", "Former", "Never"]
    age: float = Field(..., ge=0, le=120)
    bmi: float = Field(..., ge=8, le=110)
    HbA1c_level: float = Field(..., ge=2, le=20)
    glucose: float = Field(..., ge=20, le=700)
    cholesterol: float = Field(..., ge=50, le=700)
    sleep_hours: float = Field(..., ge=0, le=24)
    triglycerides: float = Field(..., ge=10, le=3000)
    physical_activity: Literal["High", "Low", "Moderate"]
    family_history: YesNo
    stress_level: Literal["Low", "Medium", "High"]
    low_hdl_cholesterol: YesNo
    high_ldl_cholesterol: YesNo
    blood_pressure: float = Field(..., ge=40, le=300)
    high_blood_pressure: YesNo
    sugar_consumption: Literal["High", "Low", "Medium"]
    crp_level: float = Field(..., ge=0, le=300)
    homocysteine_level: float = Field(..., ge=0, le=150)
    systolic_bp: float = Field(..., ge=50, le=300)
    diastolic_bp: float = Field(..., ge=20, le=200)
    alcohol_intake: float = Field(..., ge=0, le=100)
    salt_intake: float = Field(..., ge=0, le=60)
    heart_rate: float = Field(..., ge=20, le=250)
    hdl: float = Field(..., ge=5, le=200)
    ldl: float = Field(..., ge=10, le=400)
    education_level: Literal["Primary", "Secondary", "Tertiary"]
    employment_status: Literal["Employed", "Retired", "Unemployed"]


class BatchRequest(BaseModel):
    records: list[PatientFeatures] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    predicted_sublabel: str
    confidence: float = Field(..., ge=0, le=1)
    probabilities: dict[str, float]
    disease_probabilities: dict[str, float] = Field(
        ..., description="Marginal P(Diabetes), P(Heart_Disease), P(Hypertension)"
    )
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class AnomalyScoreResponse(BaseModel):
    iso_score: float = Field(
        ..., description="Isolation Forest (lower = more anomalous)"
    )
    ocsvm_score: float = Field(..., description="One-Class SVM (negative = anomalous)")
    is_anomaly_isolation: bool
    is_anomaly_svm: bool
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_version: str | None = None
