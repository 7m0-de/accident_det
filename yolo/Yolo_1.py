from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(frame, persist=True)

    annotated_frame = results[0].plot()
    boxes = results[0].boxes

    if boxes.id is not None:
        ids = boxes.id.cpu().numpy()
        coords = boxes.xyxy.cpu().numpy()

        for i in range(len(ids)):
            x1, y1, x2, y2 = coords[i]
            obj_id = int(ids[i])

            cv2.putText(
                annotated_frame,
                f"ID {obj_id}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("video", annotated_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()