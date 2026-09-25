import json
from tg_config import settings

def get_whitelist() -> list[int]:
    path = settings.whitelist_path
    if not path.exists():
        return [settings.ADMIN_ID]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [int(uid) for uid in data if str(uid).isdigit()]
    except Exception:
        return [settings.ADMIN_ID]

def is_user_allowed(user_id: int) -> bool:
    if settings.ADMIN_ID and user_id == settings.ADMIN_ID:
        return True
    return user_id in get_whitelist()
