"""
Class containing output format for LLM response
"""
from pydantic import BaseModel


class SentimentExtraction(BaseModel):
    app_features: str


class SentimentGradeExtraction(BaseModel):
    sentiment_grade: float
