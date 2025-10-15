# 진입점(ENTRY POINT)
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
import uvicorn
from fastapi import FastAPI
from core.config import configure_app
from api.routes import router as api_router
from core.json_safe import SafeJSONResponse

app = FastAPI(
    title="Korean Stock Correlation Analysis",
    description="한국 주식 시장 상관관계 분석",
    version="2.0.0",
    default_response_class=SafeJSONResponse
)

configure_app(app)
app.include_router(api_router) # 모든 엔드포인트 등록

if __name__ == "__main__":
    uvicorn.run("app", host="0.0.0.0", port=8000)