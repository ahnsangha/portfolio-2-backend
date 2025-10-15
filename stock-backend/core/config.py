# 공통 설정
# CORS, 정적 파일, Jinja2Templates 경로 등을 초기화
# configure_app(app) 한 함수만 외부에 노출
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent.parent # stock-backend/

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# 허용할 출처(Origin) 목록
origins = [
    "http://localhost:5173",  # 로컬 개발 환경 주소
    "https://portfolio-2-frontend-eight.vercel.app",  # Vercel 배포 주소
]

# CORS·static·templates 설정
def configure_app(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins, # 특정 출처만 명시적으로 허용
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    static_dir = BASE_DIR / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")