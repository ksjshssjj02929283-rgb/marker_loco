#!/usr/bin/env bash
source /opt/ros/humble/setup.bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
capture_dir="$repo_dir/captures"
mkdir -p "$capture_dir"
bag_dir="$capture_dir/usb_camera_$(date +%Y%m%d_%H%M%S)"
camera_pid=
preview_pid=

cleanup() {
  if [[ -n "$preview_pid" ]]; then
    kill "$preview_pid" 2>/dev/null || true
    wait "$preview_pid" 2>/dev/null || true
  fi
  if [[ -n "$camera_pid" ]]; then
    kill "$camera_pid" 2>/dev/null || true
    wait "$camera_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

publisher_count() {
  ros2 topic info /usb_camera/image_raw 2>/dev/null | awk '/Publisher count:/ {print $3}' || true
}

count=$(publisher_count)
if (( ${count:-0} < 1 )); then
  echo "启动 /dev/video0 (MJPEG, 1920x1080, 30 fps)..."
  ros2 run usb_cam usb_cam_node_exe --ros-args \
    -r __ns:=/usb_camera \
    -p video_device:=/dev/video0 \
    -p pixel_format:=mjpeg2rgb \
    -p image_width:=1920 -p image_height:=1080 \
    -p framerate:=30.0 \
    >"$capture_dir/usb_camera_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
  camera_pid=$!

  for _ in {1..30}; do
    count=$(publisher_count)
    if (( ${count:-0} >= 1 )); then
      break
    fi
    if ! kill -0 "$camera_pid" 2>/dev/null; then
      echo "相机启动失败，请检查相机日志和 /dev/video0 是否被占用。" >&2
      exit 1
    fi
    sleep 0.5
  done
  count=$(publisher_count)
  if (( ${count:-0} < 1 )); then
    echo "等待相机图像话题超时。" >&2
    exit 1
  fi
else
  echo "复用正在运行的 /usb_camera/image_raw 发布者。"
fi

ros2 run rqt_image_view rqt_image_view /usb_camera/image_raw &
preview_pid=$!
echo "正在录制：$bag_dir"
echo "按 Ctrl+C 停止录制。"
ros2 bag record -o "$bag_dir" \
  /usb_camera/image_raw /usb_camera/camera_info /tf /tf_static
