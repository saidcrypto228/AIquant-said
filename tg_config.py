from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    BOT_TOKEN: str = ""
    TG_BOT_TOKEN: str = ""
    ADMIN_ID: int = 7001461641
    TMA_URL: str = "http://localhost:8000"
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    DATA_DIR: str = "data"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def token(self) -> str:
        return self.BOT_TOKEN or self.TG_BOT_TOKEN

    @property
    def state_path(self) -> Path:
        return Path(self.DATA_DIR) / "bot_state.json"

    @property
    def panic_trigger_path(self) -> Path:
        return Path(self.DATA_DIR) / "panic.trigger"

    @property
    def whitelist_path(self) -> Path:
        return Path(self.DATA_DIR) / "whitelist.json"

settings = Settings()
