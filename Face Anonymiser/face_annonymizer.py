import os
import argparse
import cv2
import mediapipe as mp

def process_img(img, face_detection):
    """
    Detects faces in the given image and applies a blurring effect on detected faces.

    Parameters:
    img (numpy.ndarray): The input image.
    face_detection (mp.solutions.face_detection.FaceDetection): Mediapipe Face Detection model.

    Returns:
    numpy.ndarray: The processed image with blurred faces.
    """
    H, W, _ = img.shape  # Get image dimensions (Height, Width, Channels)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR image to RGB (required for Mediapipe)
    out = face_detection.process(img_rgb)  # Perform face detection

    if out.detections:  # Check if any face is detected
        for detection in out.detections:
            location_data = detection.location_data  # Get bounding box details
            bbox = location_data.relative_bounding_box  # Bounding box is in relative coordinates (0 to 1)

            # Extract bounding box coordinates
            x1 = int(bbox.xmin * W)
            y1 = int(bbox.ymin * H)
            w = int(bbox.width * W)
            h = int(bbox.height * H)

            # Apply blur to detected face region
            img[y1:y1 + h, x1:x1 + w] = cv2.blur(img[y1:y1 + h, x1:x1 + w], (30, 30))

    return img  # Return processed image

# Argument parser for command-line inputs
parser = argparse.ArgumentParser(description="Face Detection with Blurring")
# choice=input('enter what you wnat to blurr image, video, webcam ')

parser.add_argument("--mode", choices=['image', 'video', 'webcam'], default='webcam', help="Mode of operation: 'image', 'video', or 'webcam'")
parser.add_argument("--filePath", default=None, help="Path to the image or video file (required for 'image' or 'video' mode)")

args = parser.parse_args()

# Create an output directory if it doesn't exist
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    if args.mode == "image":
        if args.filePath is None:
            raise ValueError("File path required for image mode!")

        img = cv2.imread(args.filePath)  # Read input image
        img = process_img(img, face_detection)  # Process image (detect and blur faces)

        output_path = os.path.join(output_dir, 'output.png')
        cv2.imwrite(output_path, img)  # Save the output image
        print(f"Processed image saved at: {output_path}")

    elif args.mode == "video":
        if args.filePath is None:
            raise ValueError("File path required for video mode!")

        cap = cv2.VideoCapture(args.filePath)  # Open video file
        ret, frame = cap.read()

        if not ret:
            raise ValueError("Error reading the video file!")

        # Initialize VideoWriter to save the output
        output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'),
                                       cv2.VideoWriter_fourcc(*'MP4V'),
                                       cap.get(cv2.CAP_PROP_FPS),  # Use the original FPS of the video
                                       (frame.shape[1], frame.shape[0]))  # Frame width & height

        while ret:
            frame = process_img(frame, face_detection)  # Process frame (detect & blur faces)
            output_video.write(frame)  # Save processed frame
            ret, frame = cap.read()  # Read next frame

        cap.release()
        output_video.release()
        print("Processed video saved at: ./output/output.mp4")

    elif args.mode == "webcam":
        cap = cv2.VideoCapture(0)  # Open webcam
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam!")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image from webcam")
                break

            frame = process_img(frame, face_detection)  # Process webcam frame
            cv2.imshow('Face Blur Webcam', frame)  # Show real-time face blurring

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break

        cap.release()
        cv2.destroyAllWindows()