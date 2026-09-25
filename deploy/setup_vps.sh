#!/bin/bash
set -e

echo "=== [QVEX] БЫСТРАЯ НАСТРОЙКА UBUNTU VPS ==="

# 1. Настройка Swap на 2GB для защиты от OOM
if [ ! -f /swapfile ]; then
    echo "[*] Создание файла подкачки 2GB Swap..."
    fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    sysctl vm.swappiness=10
    echo 'vm.swappiness=10' >> /etc/sysctl.conf
    echo "[✓] Swap 2GB успешно подключен."
fi

# 2. Установка системных сервисов
echo "[*] Установка systemd сервисов..."
cp deploy/qvex-*.service /etc/systemd/system/
systemctl daemon-reload

echo "[✓] Сервисы зарегистрированы. Управление:"
echo "    systemctl start qvex-core qvex-tma qvex-tg"
echo "    systemctl enable qvex-core qvex-tma qvex-tg"
echo "    journalctl -u qvex-core -f"
