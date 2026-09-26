import json
from pathlib import Path
from typing import Any, Dict

from .base import TradingService


class QVEXTradingService(TradingService):
    """
    Real QVEX Control Plane adapter.

    Reads canonical CoreState and sends control commands through IPC.
    It does NOT import the trading engine.
    """

    def __init__(
        self,
        state_path: str,
        control_path: str,
    ):
        self.state_path = Path(state_path)
        self.control_path = Path(control_path)

    def get_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            raise FileNotFoundError(
                f"Canonical CoreState not found: {self.state_path}"
            )

        return json.loads(
            self.state_path.read_text(encoding="utf-8")
        )

    def _control_manager(self):
        import sys

        root = self.control_path.parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from control_ipc import ControlStateManager

        return ControlStateManager(
            filepath=str(self.control_path)
        )

    def pause_trading(self) -> Dict[str, Any]:
        self._control_manager().pause_trading(
            admin_tag="control_plane"
        )
        return self.get_state()

    def resume_trading(self) -> Dict[str, Any]:
        self._control_manager().resume_trading(
            admin_tag="control_plane"
        )
        return self.get_state()

    def emergency_close_all(self) -> Dict[str, Any]:
        self._control_manager().trigger_panic(
            admin_tag="control_plane"
        )
        return self.get_state()

    def close_position(self, symbol: str) -> Dict[str, Any]:
        raise NotImplementedError(
            "Per-position close requires a dedicated Core IPC command."
        )
