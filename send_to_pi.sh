#!/usr/bin/env bash

# Exit immediately if any command returns a non-zero status (error)
set -e

# Check if the user provided an IP address argument
if [ -z "$1" ]; then
  echo "Error: No IP address provided."
  echo "Usage: ./deploy.sh <IP_ADDRESS>"
  exit 1
fi

PI_IP="$1"
PI_USER="alessio"
BINARY_PATH="target/aarch64-unknown-linux-gnu/release/lspi_diagnostics"

echo "========================================="
echo "🚀 Building for Raspberry Pi (aarch64)..."
echo "========================================="
cargo build --target aarch64-unknown-linux-gnu --release --no-default-features

echo "========================================="
echo "🛑 Stopping robot.service and clearing old binary..."
echo "========================================="
# The '|| true' ensures the script doesn't crash if the service is already stopped
ssh "$PI_USER@$PI_IP" "sudo systemctl stop robot.service || true; rm -f ~/lspi_diagnostics"

echo "========================================="
echo "📦 Uploading binary to $PI_USER@$PI_IP..."
echo "========================================="
scp "$BINARY_PATH" "$PI_USER@$PI_IP:~/"

echo "========================================="
echo "🔧 Patching Nix Linker and Starting Service..."
echo "========================================="
# 1. Make executable
# 2. Patch the hardcoded Nix linker to use standard Raspberry Pi OS linker
# 3. Start the service
ssh "$PI_USER@$PI_IP" "chmod +x ~/lspi_diagnostics && patchelf --set-interpreter /lib/ld-linux-aarch64.so.1 ~/lspi_diagnostics && sudo systemctl start robot.service"

echo "========================================="
echo "✅ Deployment Successful!"
echo "========================================="
