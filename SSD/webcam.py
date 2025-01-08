import cv2
import torch
import torchvision.transforms.v2 as v2
from PIL import Image


model = torch.jit.load('./models/best.pt')
model.eval()

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

FPS = 15

cap.set(cv2.CAP_PROP_FPS, FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("webcam_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), FPS, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame. Check your camera feed.")
        break

    input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(input_image)
    input_tensor = input_tensor.to("cuda")

    with torch.no_grad():
        detections = model([input_tensor])

    for b, s, l in zip(detections[1][0]["boxes"], detections[1][0]["scores"], detections[1][0]["labels"]):
        if s > 0.3:
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Class {l}: {s:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


    cv2.imshow('Webcam Feed', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
