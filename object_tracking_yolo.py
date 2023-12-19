import cv2
import numpy as np

# Load YOLO model and configuration
net = cv2.dnn.readNet(r".\yolov3.weights", r".\yolov3.cfg")
classes = []
with open(r".\coco.names", "r") as f:
    classes = [line.strip() for line in f]
    print(classes)
# Get output layer names
layer_names = net.getUnconnectedOutLayersNames()


# Open video capture
cap = cv2.VideoCapture(r"video.mp4")
# Get the video's width, height, and frames per second (fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # You can use other codecs like 'MJPG', 'XVID', 'MP4V', etc.
out1 = cv2.VideoWriter('output_video.avi', fourcc, fps, (width, height))

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    resized_frame = cv2.resize(frame, (416, 416))

    # Normalize and prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(resized_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Run forward pass and get predictions
    outs = net.forward(layer_names)

    # Post-process the results
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:

        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.8:  # Adjust confidence threshold as needed
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply non-maximum suppression to remove redundant boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # print(class_ids)
    # Draw bounding boxes on the frame
    for i in indices:
        # if 45 in class_ids:
        box = boxes[i]
        x, y, w, h = box
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        xm = ((x) + (x+w))/2
        ym = ((y) + (y+h))/2
        # cv2.circle(frame, (int(xm),int(ym)), 2, (0, 0, 255), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,255), 2)
        cv2.putText(frame,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)

    out1.write(frame)
    # Display the resulting frame
    cv2.imshow("YOLO Object Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
out1.release()
cv2.destroyAllWindows()
