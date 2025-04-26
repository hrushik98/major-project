from inference import get_model
import supervision as sv
import cv2
import time
import os

# Set your API key
os.environ["ROBOFLOW_API_KEY"] = "78zUzBtrpgjMypEYxfHI"

try:
    # Load the model using inference library
    model = get_model(model_id="weeds-nxe1w/1")
    print("Roboflow model loaded successfully")
except Exception as e:
    print(f"Error initializing Roboflow model: {e}")
    exit(1)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

# Set properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# For FPS calculation
prev_time = time.time()
frame_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    try:
        # Run inference on the frame
        results = model.infer(frame)[0]
        
        # Convert results to supervision Detections
        detections = sv.Detections.from_inference(results)
        
        # Annotate the image with our inference results
        annotated_frame = frame.copy()
        if len(detections) > 0:
            annotated_frame = bounding_box_annotator.annotate(
                scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections)
        
        # Calculate and display FPS
        frame_count += 1
        current_time = time.time()
        if current_time - prev_time >= 1.0:
            fps = frame_count / (current_time - prev_time)
            frame_count = 0
            prev_time = current_time
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Weed Detection', annotated_frame)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        cv2.imshow('Weed Detection', frame)  # Show original frame on error
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
