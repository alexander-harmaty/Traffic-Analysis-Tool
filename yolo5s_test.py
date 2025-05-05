from ultralytics import YOLO
import cv2

# Load pre-trained YOLOv5s model
model = YOLO("yolov5s.pt")

# Load video
video_path = "videos/nyc_highway.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection on frame
    results = model.predict(frame, imgsz=640, conf=0.4, device=0)

    # Draw detections on the frame
    annotated_frame = results[0].plot()

    # Display output
    cv2.imshow("Traffic Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
