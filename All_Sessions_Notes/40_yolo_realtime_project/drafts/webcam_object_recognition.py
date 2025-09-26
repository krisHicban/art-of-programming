import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


import cv2

cap = cv2.VideoCapture(0)  # 0 = webcam
while True:
    ret, frame = cap.read()
    results = model(frame)
    cv2.imshow('YOLO Real-Time Detection', results.render()[0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()