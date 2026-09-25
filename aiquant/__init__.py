"""Alias for backward compatibility. Imports routed to qvex."""
import sys
import qvex
sys.modules["aiquant"] = qvex
