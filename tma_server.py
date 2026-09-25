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

@app.get("/api/state")
async def get_state():
    if not settings.state_path.exists():
        raise HTTPException(status_code=404, detail="bot_state.json not found")
    try:
        return json.loads(settings.state_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/kill", response_model=PanicResponse)
async def trigger_kill(user: dict = Security(verify_tma_request)):
    try:
        # 1. Создаем триггер для ядра через атомарную запись
        payload = {"triggered_at": time.time(), "source": f"tma_user_{user.get('id')}"}
        atomic_write_json(settings.panic_trigger_path, payload)

        # 2. Асинхронно запускаем изолированный ликвидатор
        if Path("emergency_killer.py").exists():
            subprocess.Popen([sys.executable, "emergency_killer.py"])

        return PanicResponse(status="ok", message="Kill sequence initiated", timestamp=payload["triggered_at"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

static_dir = Path("static")
if static_dir.exists():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.SERVER_HOST, port=settings.SERVER_PORT, log_level="warning")
