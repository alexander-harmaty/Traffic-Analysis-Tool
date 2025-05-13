import cv2
import argparse
import os
import numpy as np
from ultralytics import YOLO
from tracker.sort.sort import Sort
from tracker.line_counter import LineCounter
from metrics_logger import TrafficLog, load_video_metadata, append_log_to_excel
from collections import defaultdict

# Filter only these classes
VEHICLE_CLASSES = ['car', 'bus', 'truck', 'motorcycle']

def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Analysis Tool")
    parser.add_argument('--source', type=str, required=True, help="Path to video file")
    return parser.parse_args()

def main():
    args = parse_args()
    video_path = args.source
    filename = os.path.basename(video_path)

    # Initialize log object with filename
    log = TrafficLog(filename=filename)

    # Load and update video metadata
    metadata_path = os.path.join("videos", "video_metadata.json")
    if os.path.exists(metadata_path):
        video_metadata = load_video_metadata(metadata_path)
        log.update_from_metadata(video_metadata)

    # Load YOLOv5s model
    model = YOLO("yolov5s.pt")

    # Initialize SORT tracker
    tracker = Sort()

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Metadata logging
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = f"{width}x{height}"
    duration = frame_count / fps if fps else 0

    log.fps = round(fps, 2)
    log.total_frames = frame_count
    log.duration = round(duration, 2)
    log.resolution = resolution

    class_names = model.names

    # Define a single horizontal counting line
    y = int(height * 0.75)
    line_left_start = (0, y)
    line_left_end = (width // 2, y)
    line_right_start = (width // 2, y)
    line_right_end = (width, y)

    line_counter_a = LineCounter(line_right_start, line_right_end, direction="down")
    line_counter_b = LineCounter(line_left_start, line_left_end, direction="up")

    count_a = 0
    count_b = 0
    previous_centroids = defaultdict(lambda: None)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=640, conf=0.4, device=0)
        detections = results[0].boxes

        vehicle_boxes = []
        if detections is not None and detections.cls is not None:
            for box, cls_id in zip(detections.xyxy, detections.cls):
                class_name = class_names[int(cls_id)]
                if class_name in VEHICLE_CLASSES:
                    x1, y1, x2, y2 = map(int, box[:4])
                    vehicle_boxes.append([x1, y1, x2, y2])

        vehicle_boxes_np = np.array(vehicle_boxes)
        tracks = tracker.update(vehicle_boxes_np)

        for track in tracks:
            x1, y1, x2, y2, track_id = track.astype(int)
            centroid = ((x1 + x2) // 2, y2)

            prev = previous_centroids[track_id]
            previous_centroids[track_id] = centroid

            if prev is None:
                continue

            dy = centroid[1] - prev[1]
            if dy > 0:  # moving down (away from camera)
                if line_counter_a.check_crossing(track_id, centroid):
                    count_a += 1
            elif dy < 0:  # moving up (toward camera)
                if line_counter_b.check_crossing(track_id, centroid):
                    count_b += 1

        # Annotate frame
        annotated_frame = results[0].plot()

        # Draw counting lines
        cv2.line(annotated_frame, line_left_start, line_left_end, (255, 0, 0), 2)
        cv2.line(annotated_frame, line_right_start, line_right_end, (0, 0, 255), 2)
        cv2.putText(annotated_frame, "DirB", (line_left_start[0] + 10, line_left_start[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(annotated_frame, "DirA", (line_right_start[0] + 10, line_right_start[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw counters
        total = count_a + count_b
        cv2.putText(annotated_frame, f"DirB: {count_b}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        cv2.putText(annotated_frame, f"DirA: {count_a}", (width - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"Total: {total}", (width // 2 - 80, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Show frame
        cv2.imshow("Traffic Analysis", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Log results
    log.total_vehicle_count = int(count_a + count_b)
    log.model_used = "yolov5s.pt"
    log.tracking_algorithm = "SORT + Line Crossing"
    log.directional_flow_dirA = int(count_a)
    log.directional_flow_dirB = int(count_b)

    append_log_to_excel(log, "Traffic_Analysis_Log.xlsx")
    print(f"Log for {filename} appended to Traffic_Analysis_Log.xlsx")

if __name__ == "__main__":
    main()
