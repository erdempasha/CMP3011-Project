import cv2
import torch
import numpy as np
import torchvision.transforms.v2 as v2
from PIL import Image

# Load the trained SSD model
model = torch.jit.load('./models/best.pt')
model.eval()

# Define the transformation for the input frames
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

# Load video
video_path = "input_video.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer for output
out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(input_image)
    input_tensor = input_tensor.to("cuda")

    # Inference
    with torch.no_grad():
        detections = model([input_tensor])

    for b, s, l in zip(detections[1][0]["boxes"], detections[1][0]["scores"], detections[1][0]["labels"]):

        if s > 0.3:
            # Draw box and label
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Class {l}: {s:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


    # Write the frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
