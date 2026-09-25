import os
import sys
import time
import signal
import re
import subprocess
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [QVEX Master] {msg}", flush=True)

def start_tunnel():
    cf_exe = Path("cloudflared.exe")
    if not cf_exe.exists():
        log("cloudflared.exe не найден. Запуск локально без HTTPS-туннеля.")
        return None, ""

    log("Запуск встроенного HTTPS-туннеля Cloudflare...")
    proc = subprocess.Popen(
        [str(cf_exe.resolve()), "tunnel", "--url", "http://127.0.0.1:8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    https_url = ""
    start_t = time.time()
    while time.time() - start_t < 15:
        line = proc.stdout.readline()
        if not line:
            continue
        m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if m:
            https_url = m.group(0)
            break

    if https_url:
        log(f"[✓] Публичный URL Mini App получен: {https_url}")
        env_p = Path(".env")
        lines = []
        if env_p.exists():
            for l in env_p.read_text(encoding="utf-8").splitlines():
                if not l.startswith("TMA_URL="):
                    lines.append(l)
        lines.append(f"TMA_URL={https_url}")
        env_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.environ["TMA_URL"] = https_url
    return proc, https_url

def main():
    log("==========================================================")
    log("   QVEX v10.7: QUANTITATIVE VECTOR EXECUTION (CORE + TG + TMA) ")
    log("==========================================================")

    python_exe = sys.executable
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    tunnel_proc, public_url = start_tunnel()
    time.sleep(1.0)

    SERVICES = [
        {"name": "TMA-SERVER", "script": "tma_server.py",     "prefix": "[1/3 TMA ]"},
        {"name": "CORE-BOT",   "script": "hl_swing_bot.py",   "prefix": "[2/3 CORE]"},
        {"name": "TG-SERVICE", "script": "tg_bot/tg_bot_service.py", "prefix": "[3/3 TG  ]"},
    ]

    procs = []
    for s in SERVICES:
        if not os.path.exists(s["script"]):
            log(f"Пропуск {s['script']} (файл не найден)")
            continue
        log(f"Запуск: {s['prefix']} {s['script']}...")
        p = subprocess.Popen([python_exe, "-u", s["script"]], env=env)
        procs.append((s, p))
        time.sleep(1.2)

    log("[OK] Все системы запущены и активны.")
    if public_url:
        log(f"[INFO] Публичный адрес Mini App: {public_url}")
    log("Для остановки нажмите Ctrl + C в этом окне.")
    log("----------------------------------------------------------")

    def shutdown(signum=None, frame=None):
        log("Остановка всех процессов...")
        for s, p in procs:
            if p.poll() is None:
                p.terminate()
                try: p.wait(timeout=3)
                except subprocess.TimeoutExpired: p.kill()
        if tunnel_proc and tunnel_proc.poll() is None:
            tunnel_proc.terminate()
        log("[OK] Все процессы остановлены.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            for s, p in procs:
                status = p.poll()
                if status is not None:
                    log(f"[!] Процесс {s['name']} завершился (Код: {status}).")
                    shutdown()
            time.sleep(1.0)
    except (KeyboardInterrupt, SystemExit):
        shutdown()

if __name__ == "__main__":
    main()
