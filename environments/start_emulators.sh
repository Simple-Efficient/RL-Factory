NUM_EMULATORS=8
BASE_GRPC_PORT=8554
BASE_TELNET_PORT=5554

for i in $(seq 0 $((NUM_EMULATORS - 1))); do
  AVD_NAME="AndroidWorldAvd_$i"
  grpc_port=$((BASE_GRPC_PORT + i))
  telnet_port=$((BASE_TELNET_PORT + i * 2))
  session_name="emulator_$i"

  echo "启动 $AVD_NAME | grpc=$grpc_port | telnet=$telnet_port"

  tmux new-session -d -s $session_name \
    "/opt/android-sdk/emulator/emulator -avd $AVD_NAME \
     -port $telnet_port -grpc $grpc_port \
     -no-snapshot -no-window -gpu swiftshader_indirect \
     -no-audio -no-boot-anim -read-only"

  sleep 2
done

echo "所有模拟器已在 tmux 会话中启动。使用 tmux ls 查看。"
