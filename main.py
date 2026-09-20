from ultralytics import YOLO
import cv2
import os

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO("models/best.pt")

# -----------------------------
# Input and output
# -----------------------------
input_video = "input/football.mp4"
output_video = "output/detected_video.mp4"

os.makedirs("output", exist_ok=True)

# -----------------------------
# Open video
# -----------------------------
cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# -----------------------------
# Video writer
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

frame_number = 0

# -----------------------------
# Process video
# -----------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # YOLO tracking
    results = model.track(
        frame,
        persist=True,
        conf=0.35,
        verbose=False
    )

    # Draw detections
    annotated_frame = results[0].plot()

    # Frame counter
    cv2.putText(
        annotated_frame,
        f"Frame: {frame_number}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display
    cv2.imshow(
        "Football Vision AI",
        annotated_frame
    )

    # Save
    out.write(annotated_frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# Release
# -----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing completed!")
print("Output saved at:", output_video)
