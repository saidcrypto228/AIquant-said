"""
aiquant/utils/console.py
========================
Shared ANSI colour helpers for CLI output. Imported by run.py and the ML
ensemble pipeline so the two stay visually consistent with one definition.
"""
import platform


def _c(code, text):
    if platform.system() == 'Windows':
        return text
    return f'\033[{code}m{text}\033[0m'


BOLD   = lambda t: _c('1',  t)
DIM    = lambda t: _c('2',  t)
GREEN  = lambda t: _c('32', t)
RED    = lambda t: _c('31', t)
CYAN   = lambda t: _c('36', t)
YELLOW = lambda t: _c('33', t)
WHITE  = lambda t: _c('97', t)
