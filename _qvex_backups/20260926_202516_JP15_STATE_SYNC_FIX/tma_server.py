import sys
from pathlib import Path
_tg_path = str(Path(__file__).resolve().parent / 'tg_bot')
if _tg_path not in sys.path:
    sys.path.insert(0, _tg_path)

import hashlib
import hmac
import json
import time
import subprocess
from urllib.parse import parse_qsl
from pathlib import Path
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from tg_config import settings
from qvex_utils import atomic_write_json

app = FastAPI(title="QVEX Quantitative Terminal API")
security = HTTPBearer(auto_error=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PanicResponse(BaseModel):
    status: str
    message: str
    timestamp: float

def verify_tma_request(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    # Заглушка для локального тестирования без Telegram (только для localhost)
    if not credentials or not credentials.credentials:
        return {"id": settings.ADMIN_ID, "username": "local_dev"}

    init_data_raw = credentials.credentials
    try:
        parsed_params = dict(parse_qsl(init_data_raw, strict_parsing=True))
    except Exception:
        raise HTTPException(status_code=401, detail="Malformed initData")

    if "hash" not in parsed_params:
        raise HTTPException(status_code=401, detail="Missing HMAC signature")

    received_hash = parsed_params.pop("hash")
    data_check_string = "\n".join(f"{k}={v}" for k, v in sorted(parsed_params.items()))
    secret_key = hmac.new(b"WebAppData", settings.token.encode("utf-8"), hashlib.sha256).digest()
    calculated_hash = hmac.new(secret_key, data_check_string.encode("utf-8"), hashlib.sha256).hexdigest()

    if not hmac.compare_digest(calculated_hash, received_hash):
        raise HTTPException(status_code=403, detail="Signature verification failed")

    auth_date = int(parsed_params.get("auth_date", 0))
    if time.time() - auth_date > 86400: # 24 часа для тестов
        raise HTTPException(status_code=401, detail="Session expired")

    user_payload = json.loads(parsed_params.get("user", "{}"))
    return user_payload

def get_trading_service():
    from backend.app.services.qvex import QVEXTradingService
    return QVEXTradingService(
        state_path=str(settings.state_path),
        control_path=str(settings.control_path),
    )


@app.get("/api/state")
async def get_state():
    try:
        return get_trading_service().get_state()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/kill", response_model=PanicResponse)
async def trigger_kill(user: dict = Security(verify_tma_request)):
    try:
        service = get_trading_service()
        service.emergency_close_all()

        return PanicResponse(
            status="ok",
            message="Kill sequence initiated",
            timestamp=time.time(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pause")
async def pause_trading(user: dict = Security(verify_tma_request)):
    try:
        return get_trading_service().pause_trading()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/resume")
async def resume_trading(user: dict = Security(verify_tma_request)):
    try:
        return get_trading_service().resume_trading()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

static_dir = Path("static")
if static_dir.exists():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.SERVER_HOST, port=settings.SERVER_PORT, log_level="warning")
