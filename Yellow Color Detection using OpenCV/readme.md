# Yellow Color Detection using OpenCV

## 📌 Introduction
This project detects **yellow color** in real-time using OpenCV and Python. It uses **HSV color space thresholding** to isolate the yellow color and draw a bounding box around detected areas in a webcam feed.

## 🔬 Scientific Concepts Behind the Project
### 1️⃣ Digital Image Representation
Images are stored as matrices of pixel values. Each pixel has intensity values in different color channels:
- **RGB (Red, Green, Blue)** – Common for display.
- **BGR (Blue, Green, Red)** – Used by OpenCV.
- **HSV (Hue, Saturation, Value)** – Useful for color-based filtering.

### 2️⃣ Why HSV Instead of RGB?
HSV separates color information (**Hue**) from intensity (**Value**) and saturation. This makes it robust against lighting changes.
- **Hue**: Defines color type (0°-360°).
- **Saturation**: Intensity of color.
- **Value**: Brightness.

### 3️⃣ Thresholding with inRange()
The function `cv2.inRange()` isolates pixels within a given HSV range. We define a lower and upper threshold for yellow.

---
## 🛠️ Installation
Ensure you have Python installed, then run:
```bash
pip install opencv-python numpy Pillow
```

---
## 📂 Project Structure
```
Yellow-Color-Detection/
│── main.py          # Main script to detect yellow color
│── util.py          # Helper functions (color range calculation)
│── requirements.txt # Dependencies
│── README.md        # Project documentation
```

---
## 📜 Code Explanation
### 📌 `util.py` – Color Range Helper
```python
import cv2
import numpy as np

def get_limits(color):
    c = np.uint8([[[color]]])  # Convert color to numpy array
    hsv_color = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)  # Convert to HSV
    lower_limit = np.array([hsv_color[0][0][0] - 10, 100, 100], dtype=np.uint8)
    upper_limit = np.array([hsv_color[0][0][0] + 10, 255, 255], dtype=np.uint8)
    return lower_limit, upper_limit
```

### 📌 `main.py` – Main Detection Script
```python
import cv2
import numpy as np
from PIL import Image
from util import get_limits

# Define yellow color in BGR
yellow_bgr = (0, 255, 255)
lower_limit, upper_limit = get_limits(yellow_bgr)

# Start webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_limit, upper_limit)
    
    mask_ = Image.fromarray(mask)
    bbox = mask_.getbbox()

    if bbox:
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
    
    cv2.imshow("Yellow Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---
## 🚀 Running the Project
1️⃣ Clone the repository:
```bash
git clone https://github.com/yourusername/Yellow-Color-Detection.git
cd Yellow-Color-Detection
```

2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

3️⃣ Run the script:
```bash
python main.py
```

**Press `q` to exit the webcam feed.**

---
## 🛠️ How It Works
1. **Captures video frames from webcam.**
2. **Converts each frame to HSV color space.**
3. **Applies thresholding to detect yellow color.**
4. **Finds the bounding box for yellow regions.**
5. **Draws a rectangle around detected areas.**
6. **Displays the video with detection in real-time.**

---
## 🎯 Possible Enhancements
- ✅ Detect multiple colors at once.
- ✅ Improve accuracy with adaptive thresholding.
- ✅ Use deep learning (YOLO, SSD) for better object detection.

---
## 🏆 Conclusion
This project gives a hands-on introduction to **color-based object detection** using OpenCV. By understanding image processing, thresholding, and HSV conversion, you can extend this to detect different objects based on color!

---
## 🤝 Contributing
Feel free to fork this repository and submit pull requests.

💡 **Happy Coding!** 🚀

