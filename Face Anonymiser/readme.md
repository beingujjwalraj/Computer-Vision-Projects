# Face Blur using MediaPipe

## 📌 Overview
This project detects faces in images, videos, or real-time webcam feed and applies a blurring effect to them using **MediaPipe** and **OpenCV**. It is a simple implementation for privacy-focused applications. It supports multiple input modes: **Image Processing**, **Video Processing**, and **Live Webcam Processing**. The processed image/video with blurred faces will be saved in the `output/` directory.

## 🚀 Features
- Detects faces and applies blur for privacy protection.
- Works with images, videos, and real-time webcam.
- Outputs processed media with blurred faces.

## 🛠 Requirements
Ensure **Python 3.7+** is installed. Install dependencies using:

```bash
pip install -r requirements.txt
📌 Dependencies:
	•	OpenCV (cv2)
	•	MediaPipe
	•	NumPy
```
## 📂 Project Structure
```
📂 Face-Blur
│── face_blur.py        # Main script for face detection and blurring
│── requirements.txt     # List of dependencies
│── README.md            # Documentation
│── output/              # Stores processed images and videos
```

##▶️ Usage

Run the script from the command line:
```
python face_blur.py --mode image --filePath "path/to/image.jpg"
```
```
python face_blur.py --mode video --filePath "path/to/video.mp4"
```
```
python face_blur.py --mode webcam
```
- **For image mode, the processed image will be saved in output/.**
- **For video mode, the processed video will be saved in output/.**
- **For webcam mode, press q to exit the real-time face blurring.**





