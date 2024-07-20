import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load MobileNetV2 model with manually downloaded weights
model = MobileNetV2(weights='C:/Users/pc/Downloads/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5')

# Define mapping for specific labels to more general terms
label_mapping = {
    'hog': 'pig',
    'boar': 'pig',
   'sow': 'pig',
    'African_elephant': 'elephant',
    'Indian_elephant': 'elephant',
}

# Load YOLO for human detection
yolo_net = cv2.dnn.readNet("C:/Users/pc/Desktop/detection/yolov3.weights", "C:/Users/pc/Desktop/detection/yolov3.cfg")
layer_names = yolo_net.getLayerNames()
try:
    unconnected_out_layers = yolo_net.getUnconnectedOutLayers()
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]
except Exception as e:
    print(f"Error while loading YOLO layers: {e}")
    exit()

classes = open("C:/Users/pc/Desktop/detection/coco.names").read().strip().split("\n")

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Human detection with YOLO
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    yolo_outs = yolo_net.forward(output_layers)

    for out in yolo_outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Human", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Resize frame to match MobileNetV2 input size
    resized_frame = cv2.resize(frame, (224, 224))

    # Preprocess the frame for prediction
    img_array = np.expand_dims(resized_frame, axis=0)
    img_array = preprocess_input(img_array)

    # Predict frame details
    predictions = model.predict(img_array)

    # Decode the predictions
    decoded_predictions = decode_predictions(predictions)

    # Map predictions to more general terms
    mapped_predictions = []
    for pred in decoded_predictions[0]:
        label = pred[1]
        if label in label_mapping:
            label = label_mapping[label]
            mapped_predictions.append((pred[0], label, pred[2]))

    # Modify the prediction part of the code to draw a box around the animal and label it
    if mapped_predictions:
        highest_confidence_prediction = max(mapped_predictions, key=lambda x: x[2])
        label = highest_confidence_prediction[1]
        confidence = highest_confidence_prediction[2]

        # Draw a rectangle around the animal
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
        # Add a label on the box
        label_text = f"{label}: {confidence:.2f}"
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Live Feed', frame)


    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()



