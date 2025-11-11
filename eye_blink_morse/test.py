from ultralytics import YOLO

# Load your custom-trained YOLOv8 model
model = YOLO('best.pt')

# Run inference on the webcam (source=0)
# 'stream=True' is recommended for live feeds for better performance
# 'show=True' will display the annotated video feed in a new window
try:
    results = model.predict(source=0, stream=True, show=True)

    # We need to iterate over the results generator to process the stream
    for r in results:
        boxes = r.boxes
        if boxes:
            for box in boxes:
                class_id = int(box.cls[0])
                print(f"Detected class: {model.names[class_id]}")

        pass

except KeyboardInterrupt:
    # Handle user interruption (e.g., pressing Ctrl+C in the terminal)
    print("Stopping the live feed...")

print("Live feed ended.")