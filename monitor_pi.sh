#!/usr/bin/env bash

# Exit immediately if any command returns a non-zero status
set -e

# Check if the user provided an IP address argument
if [ -z "$1" ]; then
  echo "Error: No IP address provided."
  echo "Usage: ./monitor.sh <IP_ADDRESS>"
  exit 1
fi

PI_IP="$1"
PI_USER="alessio"
SERVICE_NAME="robot.service"

echo "========================================="
echo "📡 Connecting to $PI_USER@$PI_IP..."
echo "========================================="
echo "Live streaming logs for $SERVICE_NAME."
echo "(Press Ctrl+C to stop monitoring)"
echo "-----------------------------------------"

# The '-t' flag forces SSH to allocate a pseudo-terminal.
# This is required so that when you press Ctrl+C, it cleanly stops journalctl
# and closes the SSH connection without leaving ghost processes on the Pi.
#
# -u : filters exactly to your robot service
# -f : 'follows' the log live
# -n 20 : shows the last 20 lines of context before streaming the live data
ssh -t "$PI_USER@$PI_IP" "sudo journalctl -u $SERVICE_NAME -f -n 20"
