import json
import os
import tempfile
from pathlib import Path

def atomic_write_json(file_path: Path | str, data: dict) -> None:
    path = Path(file_path)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", dir=directory, delete=False, encoding="utf-8") as tf:
        json.dump(data, tf, ensure_ascii=False, indent=2)
        tf.flush()
        os.fsync(tf.fileno())
        temp_name = tf.name

    os.replace(temp_name, path)
