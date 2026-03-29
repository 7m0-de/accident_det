from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("video.mp4")

previous_positions = {}
while True:
  object_data = {}
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
          center_x = int((x1 + x2) / 2)
          center_y = int((y1 + y2) / 2)

          cv2.putText(
              annotated_frame,
              f"ID {obj_id}",
              (int(x1), int(y1) - 10),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.6,
              (0, 255, 0),
              2
          )
          dx, dy = 0, 0
          speed = (dx**2 + dy**2)**0.5
          if obj_id in previous_positions:
            prev_x, prev_y = previous_positions[obj_id]
            dx = center_x - prev_x
            dy = center_y - prev_y
          object_data[obj_id] = {
                "position": (center_x, center_y),
                "movement": (dx, dy),
                "speed": speed
            }
          previous_positions[obj_id] = (center_x, center_y)
      for id1 in object_data:
        for id2 in object_data:
          if id1 == id2:
            continue
          x1, y1 = object_data[id1]["position"]
          x2, y2 = object_data[id2]["position"]
          distance = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
          print(f"Distance between {id1} and {id2}: {distance}")
              
  print(object_data)
  cv2.imshow("video", annotated_frame)

  if cv2.waitKey(25) & 0xFF == ord('q'):
     break

cap.release()
cv2.destroyAllWindows()