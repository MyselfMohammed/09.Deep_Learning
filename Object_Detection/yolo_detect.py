import os
import sys
import argparse
import glob
import time
import datetime
import cv2
import numpy as np
from ultralytics import YOLO

# ------------------------- Argument Parsing -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model (.pt)')
parser.add_argument('--source', required=True, help='Input source: image/video/folder/cam index')
parser.add_argument('--thresh', type=float, default=0.5, help='Confidence threshold')
parser.add_argument('--resolution', default=None, help='Resolution WxH (e.g. 1280x720)')
parser.add_argument('--record', action='store_true', help='Record output to a video file')
args = parser.parse_args()

# ------------------------- Input Values -------------------------
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# ------------------------- Validate Model -------------------------
if not os.path.exists(model_path):
    print(f"[❌ ERROR] Model not found: {model_path}")
    sys.exit(1)

# ------------------------- Load Model -------------------------
model = YOLO(model_path, task='detect')
labels = model.names

# ------------------------- Determine Input Source -------------------------
img_ext_list = ['.jpg', '.jpeg', '.png', '.bmp']
vid_ext_list = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext.lower() in img_ext_list:
        source_type = 'image'
    elif ext.lower() in vid_ext_list:
        source_type = 'video'
    else:
        print(f"[❌ ERROR] Unsupported file extension: {ext}")
        sys.exit(1)
elif img_source.isdigit():
    source_type = 'usb'
    usb_idx = int(img_source)
else:
    print(f"[❌ ERROR] Invalid input source: {img_source}")
    sys.exit(1)

# ------------------------- Resolution Setup -------------------------
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.lower().split('x'))

# ------------------------- Recording Setup -------------------------
if record:
    if source_type not in ['video', 'usb']:
        print("[❌ ERROR] --record is only supported for video/camera sources.")
        sys.exit(1)
    if not user_res:
        print("[❌ ERROR] Please specify --resolution when recording.")
        sys.exit(1)

    # Save video to the same directory as the script
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    record_path = os.path.join(output_dir, f"inference_{timestamp}.avi")
    recorder = cv2.VideoWriter(record_path, cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

# ------------------------- Input Capture -------------------------
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [img for img in glob.glob(os.path.join(img_source, '*')) if os.path.splitext(img)[1].lower() in img_ext_list]
elif source_type == 'video':
    cap = cv2.VideoCapture(img_source)
    if resize:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'usb':
    cap = cv2.VideoCapture(usb_idx)
    if resize:
        cap.set(3, resW)
        cap.set(4, resH)

# ------------------------- BBox Colors -------------------------
bbox_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
               (255, 255, 0), (255, 0, 255), (0, 255, 255)]

# ------------------------- Inference Loop -------------------------
fps_list = []
img_count = 0

while True:
    t_start = time.perf_counter()

    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print("[✅ DONE] All images processed.")
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    else:
        ret, frame = cap.read()
        if not ret:
            print("[✅ DONE] Stream ended or camera disconnected.")
            break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0

    for det in detections:
        conf = det.conf.item()
        if conf < min_thresh:
            continue

        xmin, ymin, xmax, ymax = map(int, det.xyxy.cpu().numpy().squeeze())
        classid = int(det.cls.item())
        label = f"{labels[classid]}: {int(conf * 100)}%"
        color = bbox_colors[classid % len(bbox_colors)]

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, label, (xmin, max(ymin - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        object_count += 1

    # FPS Calculation
    t_end = time.perf_counter()
    fps = 1 / (t_end - t_start)
    fps_list.append(fps)
    if len(fps_list) > 100:
        fps_list.pop(0)
    avg_fps = sum(fps_list) / len(fps_list)

    # Overlay Info
    cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f"Objects: {object_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # Show & Save Frame
    display_frame = cv2.resize(frame, (1280, 960))  # For clearer display
    cv2.imshow("YOLOv8 Detection", display_frame)
    if record:
        recorder.write(frame)

    key = cv2.waitKey(5)
    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        cv2.imwrite(os.path.join(output_dir, 'capture.jpg'), frame)

# ------------------------- Cleanup -------------------------
if source_type in ['video', 'usb']:
    cap.release()
if record:
    recorder.release()
    print(f"[💾 SAVED] Inference video saved to: {record_path}")
cv2.destroyAllWindows()
