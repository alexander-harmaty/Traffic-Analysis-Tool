import cv2
import argparse
import os
import numpy as np
from ultralytics import YOLO
from tracker.sort.sort import Sort
from tracker.line_counter import LineCounter
from metrics_logger import TrafficLog, load_video_metadata, append_log_to_excel
from collections import defaultdict
import time

# Filter only these classes
VEHICLE_CLASSES = ['car', 'bus', 'truck', 'motorcycle']


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Analysis Tool")
    parser.add_argument('--source', type=str, required=True, help="Path to video file")
    return parser.parse_args()


# Global variable to store the calibration line endpoints
calibration_line = []

def click_event(event, x, y, flags, param):
    global calibration_line
    if event == cv2.EVENT_LBUTTONDOWN:
        calibration_line.append((x, y))
        if len(calibration_line) == 2:
            cv2.line(param, calibration_line[0], calibration_line[1], (0, 255, 255), 2)
            cv2.imshow("Calibration", param)

def calibrate_pixel_to_meter_ratio(frame):
    global calibration_line
    clone = frame.copy()
    cv2.imshow("Calibration", clone)
    cv2.setMouseCallback("Calibration", click_event, clone)
    print("Draw a line across a single lane. Press any key when done.")
    cv2.waitKey(0)
    cv2.destroyWindow("Calibration")

    if len(calibration_line) == 2:
        lane_width_pixels = np.linalg.norm(np.array(calibration_line[0]) - np.array(calibration_line[1]))
        return lane_width_pixels / 3.35  # Convert 11 feet (3.35m) to pixels
    else:
        print("Calibration failed. Using default ratio of 1.0")
        return 1.0


def main():
    args = parse_args()
    video_path = args.source
    filename = os.path.basename(video_path)

    log = TrafficLog(filename=filename)
    metadata_path = os.path.join("videos", "video_metadata.json")
    if os.path.exists(metadata_path):
        video_metadata = load_video_metadata(metadata_path)
        log.update_from_metadata(video_metadata)

    model = YOLO("yolov5s.pt")
    tracker = Sort()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

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

    # Read first frame for calibration
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    pixel_to_meter_ratio = calibrate_pixel_to_meter_ratio(first_frame)

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
    class_counts_a = defaultdict(int)
    class_counts_b = defaultdict(int)

    track_start_times = {}
    track_start_positions = {}
    speeds = []
    speeds_a = []
    speeds_b = []
    class_speeds_a = defaultdict(list)
    class_speeds_b = defaultdict(list)

    frame_index = 0
    inference_times = []
    read_frames = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        read_frames += 1
        start_time = time.time()
        results = model.predict(frame, imgsz=640, conf=0.4, device=0)
        end_time = time.time()
        inference_times.append((end_time - start_time) * 1000)  # ms

        detections = results[0].boxes

        vehicle_boxes = []
        box_classes = []
        if detections is not None and detections.cls is not None:
            for box, cls_id in zip(detections.xyxy, detections.cls):
                class_name = class_names[int(cls_id)]
                if class_name in VEHICLE_CLASSES:
                    x1, y1, x2, y2 = map(int, box[:4])
                    vehicle_boxes.append([x1, y1, x2, y2])
                    box_classes.append(class_name)

        vehicle_boxes_np = np.array(vehicle_boxes)
        tracks = tracker.update(vehicle_boxes_np)

        for i, track in enumerate(tracks):
            x1, y1, x2, y2, track_id = track.astype(int)
            centroid = ((x1 + x2) // 2, y2)
            prev = previous_centroids[track_id]
            previous_centroids[track_id] = centroid
            if prev is None:
                continue

            dy = centroid[1] - prev[1]
            class_name = box_classes[i] if i < len(box_classes) else "unknown"

            if track_id not in track_start_times:
                track_start_times[track_id] = frame_index
                track_start_positions[track_id] = centroid

            if dy > 0:
                if line_counter_a.check_crossing(track_id, centroid):
                    count_a += 1
                    class_counts_a[class_name] += 1
                    start_frame = track_start_times.get(track_id, frame_index)
                    start_pos = track_start_positions.get(track_id, centroid)
                    frame_delta = frame_index - start_frame
                    if frame_delta > 10:
                        pixel_distance = abs(centroid[1] - start_pos[1])
                        meters = pixel_distance / pixel_to_meter_ratio
                        seconds = frame_delta / fps
                        speed = meters / seconds * 2.23694
                        speeds.append(speed)
                        speeds_a.append(speed)
                        class_speeds_a[class_name].append(speed)

            elif dy < 0:
                if line_counter_b.check_crossing(track_id, centroid):
                    count_b += 1
                    class_counts_b[class_name] += 1
                    start_frame = track_start_times.get(track_id, frame_index)
                    start_pos = track_start_positions.get(track_id, centroid)
                    frame_delta = frame_index - start_frame
                    if frame_delta > 10:
                        pixel_distance = abs(centroid[1] - start_pos[1])
                        meters = pixel_distance / pixel_to_meter_ratio
                        seconds = frame_delta / fps
                        speed = meters / seconds * 2.23694
                        speeds.append(speed)
                        speeds_b.append(speed)
                        class_speeds_b[class_name].append(speed)

        frame_index += 1

        annotated_frame = results[0].plot()
        cv2.line(annotated_frame, line_left_start, line_left_end, (255, 0, 0), 2)
        cv2.line(annotated_frame, line_right_start, line_right_end, (0, 0, 255), 2)
        cv2.putText(annotated_frame, "DirB", (line_left_start[0] + 10, line_left_start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(annotated_frame, "DirA", (line_right_start[0] + 10, line_right_start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        total = count_a + count_b
        cv2.putText(annotated_frame, f"DirB: {count_b}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        cv2.putText(annotated_frame, f"DirA: {count_a}", (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"Total: {total}", (width // 2 - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Traffic Analysis", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    log.inference_time_per_frame_ms = round(sum(inference_times) / len(inference_times), 2) if inference_times else None
    log.average_fps = round(read_frames / duration, 2) if duration else None
    log.frame_drop_count = frame_count - read_frames

    log.total_vehicle_count = total
    log.dirA_vehicle_count = int(count_a)
    log.dirB_vehicle_count = int(count_b)
    log.car_count = class_counts_a['car'] + class_counts_b['car']
    log.truck_count = class_counts_a['truck'] + class_counts_b['truck']
    log.bus_count = class_counts_a['bus'] + class_counts_b['bus']
    log.motorcycle_count = class_counts_a['motorcycle'] + class_counts_b['motorcycle']

    log.dirA_car_count = class_counts_a['car']
    log.dirA_truck_count = class_counts_a['truck']
    log.dirA_bus_count = class_counts_a['bus']
    log.dirA_motorcycle_count = class_counts_a['motorcycle']

    log.dirB_car_count = class_counts_b['car']
    log.dirB_truck_count = class_counts_b['truck']
    log.dirB_bus_count = class_counts_b['bus']
    log.dirB_motorcycle_count = class_counts_b['motorcycle']

    minutes = duration / 60 if duration else 0
    if minutes > 0:
        log.vpm_total = round(total / minutes, 2)
        log.vpm_dirA = round(count_a / minutes, 2)
        log.vpm_dirB = round(count_b / minutes, 2)

    area = width * height
    if area > 0:
        log.congestion_density_total = round(total / area * 100000, 4)
        log.congestion_density_dirA = round(count_a / area * 100000, 4)
        log.congestion_density_dirB = round(count_b / area * 100000, 4)

    if speeds:
        log.overall_avg_speed = round(sum(speeds) / len(speeds), 2)
    if speeds_a:
        log.avg_speed_dirA = round(sum(speeds_a) / len(speeds_a), 2)
    if speeds_b:
        log.avg_speed_dirB = round(sum(speeds_b) / len(speeds_b), 2)

    log.avg_car_speed_dirA = round(sum(class_speeds_a['car']) / len(class_speeds_a['car']), 2) if class_speeds_a['car'] else 0
    log.avg_truck_speed_dirA = round(sum(class_speeds_a['truck']) / len(class_speeds_a['truck']), 2) if class_speeds_a['truck'] else 0
    log.avg_bus_speed_dirA = round(sum(class_speeds_a['bus']) / len(class_speeds_a['bus']), 2) if class_speeds_a['bus'] else 0
    log.avg_motorcycle_speed_dirA = round(sum(class_speeds_a['motorcycle']) / len(class_speeds_a['motorcycle']), 2) if class_speeds_a['motorcycle'] else 0

    log.avg_car_speed_dirB = round(sum(class_speeds_b['car']) / len(class_speeds_b['car']), 2) if class_speeds_b['car'] else 0
    log.avg_truck_speed_dirB = round(sum(class_speeds_b['truck']) / len(class_speeds_b['truck']), 2) if class_speeds_b['truck'] else 0
    log.avg_bus_speed_dirB = round(sum(class_speeds_b['bus']) / len(class_speeds_b['bus']), 2) if class_speeds_b['bus'] else 0
    log.avg_motorcycle_speed_dirB = round(sum(class_speeds_b['motorcycle']) / len(class_speeds_b['motorcycle']), 2) if class_speeds_b['motorcycle'] else 0

    log.avg_car_speed = round((log.avg_car_speed_dirA + log.avg_car_speed_dirB) / 2, 2)
    log.avg_truck_speed = round((log.avg_truck_speed_dirA + log.avg_truck_speed_dirB) / 2, 2)
    log.avg_bus_speed = round((log.avg_bus_speed_dirA + log.avg_bus_speed_dirB) / 2, 2)
    log.avg_motorcycle_speed = round((log.avg_motorcycle_speed_dirA + log.avg_motorcycle_speed_dirB) / 2, 2)

    log.model_used = "yolov5s.pt"
    log.tracking_algorithm = "SORT + Line Crossing"

    append_log_to_excel(log, "Traffic_Analysis_Log.xlsx")
    print(f"Log for {filename} appended to Traffic_Analysis_Log.xlsx")


if __name__ == "__main__":
    main()
