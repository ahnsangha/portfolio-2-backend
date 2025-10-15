# 공통 설정
# CORS, 정적 파일, Jinja2Templates 경로 등을 초기화
# configure_app(app) 한 함수만 외부에 노출
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent.parent # stock-backend/

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# CORS·static·templates 설정
def configure_app(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    static_dir = BASE_DIR / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
