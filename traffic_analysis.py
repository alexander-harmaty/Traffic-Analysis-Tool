import cv2
import argparse
import os
from ultralytics import YOLO
from tracker.centroid_tracker import CentroidTracker
from metrics_logger import TrafficLog, load_video_metadata, append_log_to_excel

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

    # Initialize tracker and total vehicle counter
    tracker = CentroidTracker(max_distance=50)
    total_vehicle_count = 0

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    resolution = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    duration = frame_count / fps if fps else 0

    log.fps = round(fps, 2)
    log.total_frames = frame_count
    log.duration = round(duration, 2)
    log.resolution = resolution

    class_names = model.names  # List of class names by index

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model.predict(frame, imgsz=640, conf=0.4, device=0)
        detections = results[0].boxes

        # Extract bounding boxes for vehicle classes
        vehicle_boxes = []
        if detections is not None and detections.cls is not None:
            for box, cls_id in zip(detections.xyxy, detections.cls):
                class_name = class_names[int(cls_id)]
                if class_name in VEHICLE_CLASSES:
                    x1, y1, x2, y2 = map(int, box[:4])
                    vehicle_boxes.append((x1, y1, x2, y2))

        # Update tracker and count new unique vehicles
        tracked_objects = tracker.update(vehicle_boxes)
        for object_id in tracked_objects:
            if object_id not in tracker.counted_ids:
                tracker.counted_ids.add(object_id)
                total_vehicle_count += 1

        # Annotated frame with bounding boxes
        annotated_frame = results[0].plot()

        # Draw total vehicle count overlay
        overlay_text = f"Total Vehicles: {total_vehicle_count}"
        cv2.putText(annotated_frame, overlay_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Display
        cv2.imshow("Traffic Analysis", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Fill log with end-of-run data
    log.total_vehicle_count = total_vehicle_count
    log.model_used = "yolov5s.pt"
    log.tracking_algorithm = "Centroid Tracker"

    # Append log to Excel
    excel_path = "Traffic_Analysis_Log.xlsx"
    append_log_to_excel(log, excel_path)
    print(f"Log for {filename} appended to {excel_path}")

if __name__ == "__main__":
    main()