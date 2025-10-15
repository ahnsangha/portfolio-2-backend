# ■ Pydantic 모델(MODELS)
#   - API 요청/응답(AnalysisRequest, AnalysisStatus) 스키마 정의.
#   - 데이터 검증·문서화 목적.

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class AnalysisRequest(BaseModel):
    start_date: str = Field(default="2023-01-01")
    end_date: str   = Field(default="2024-12-31")
    tickers: Optional[List[str]] = None
    window: int = 60

class AnalysisStatus(BaseModel):
    task_id: str
    status: str
    message: str
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
