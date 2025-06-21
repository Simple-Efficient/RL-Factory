#!/bin/bash

# system-images;android-33;google_apis;x86_64
PACKAGE="system-images;android-33;google_apis;x86_64"
DEVICE="pixel_6"
SDCARD_SIZE="512M"

# 创建 8 个 AVD
for i in $(seq 0 7); do
  AVD_NAME="AndroidWorldAvd_$i"

  echo "Creating AVD: $AVD_NAME"

  echo "no" | avdmanager create avd \
    --name "$AVD_NAME" \
    --device "$DEVICE" \
    --package "$PACKAGE" \
    --sdcard "$SDCARD_SIZE" \
    --force

  echo " Created $AVD_NAME"
done
