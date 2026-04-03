# Golf Bot System: AI Tracking, Distance Sensing & Club Assistant ⛳🤖

## 📝 Project Description
A smart golf bot system combining autonomous ball tracking and real-time data analysis. Built with Python, YOLOv8, and ESP32-CAM, it features a live-feed dashboard for robot control, an animated vision-based distance gauge (10-30cm), and a club assistant that calculates remaining yardage to recommend the optimal professional golf club.

## 🚀 Key Features
1.  **Centralized Home Dashboard:** Modern glassmorphism-style UI for quick access to all bot modules.
2.  **Interactive Navigation:** Seamless switching between Steering, Distance, and Club Recommendation panels.
3.  **Real-Time AI Stream:** Low-latency ESP-CAM live feed with integrated AI HUD diagnostics.
4.  **Multi-Input Control:** Support for virtual D-Pad and physical Keyboard (WASD/Arrows) steering.
5.  **Autonomous Tracking:** AI-powered object following using YOLOv8 machine learning models.
6.  **Dynamic Speed Control:** Real-time throttle adjustments for precise robot movement.
7.  **Smart Distance Gauge:** Vision-based measurement system for objects within a 10cm–30cm range.
8.  **Measurement Lock:** Automatically captures and freezes stable distance data upon detection.
9.  **Live Telemetry:** Dashboard for reporting detections, FPS, and real-time server logs.
10. **Quick Reset:** One-click functionality to clear measurements and restart ball scanning.
11. **Club Assistant:** Intelligent yardage calculator and professional club recommendation engine.

## 🛠️ Tech Stack
- **Backend:** Python, FastAPI, Uvicorn
- **Frontend:** Modern HTML5, CSS3, Vanilla JavaScript
- **AI/Vision:** Ultralytics YOLOv8, OpenCV (cv2)
- **Hardware:** ESP32-CAM (MJPEG Streamer), HTTP/WebSocket protocols

## ⚙️ How to Run
1. **Navigate to the server folder:**
   ```bash
   cd 8510_repo_multi
   ```
2. **Install requirements:**
   *(Ensure you have Python installed)*
   ```bash
   pip install fastapi uvicorn opencv-python ultralytics httpx pyyaml
   ```
3. **Run the server:**
   ```bash
   python server.py
   ```
4. **Access the interface:**
   Open `http://localhost:8000` in your web browser.

---
**Developed for the Golf Bot Intelligent Hardware Suite.**
