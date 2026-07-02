#!/usr/bin/env bash

# Exit immediately if any command returns a non-zero status
set -e

# Check if the user provided an IP address argument
if [ -z "$1" ]; then
  echo -e "\x1b[31mError: No IP address provided.\x1b[0m"
  echo "Usage: ./monitor.sh <IP_ADDRESS>"
  exit 1
fi

PI_IP="$1"
PI_USER="alessio"
SERVICE_NAME="robot.service"

echo "========================================="
echo -e "\x1b[36m📡 Connecting to $PI_USER@$PI_IP...\x1b[0m"
echo "========================================="
echo "Live streaming logs for $SERVICE_NAME."
echo "(Press Ctrl+C to stop monitoring)"
echo "-----------------------------------------"

# Replaced '--color=always' with 'SYSTEMD_COLORS=1'
ssh -t "$PI_USER@$PI_IP" "sudo SYSTEMD_COLORS=1 journalctl -u $SERVICE_NAME -f -n 20"
