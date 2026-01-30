# ESP32 Camera Setup Guide

## 1. ESP32 Hardware Setup
- Use ESP32-CAM module (AI-Thinker)
- Connect USB programmer for flashing
- Ensure camera is properly connected

## 2. ESP32 Code Configuration
1. Open `esp32_camera.ino` in Arduino IDE
2. Install required libraries:
   - ESP32 Camera library
   - WiFi library (built-in)
3. Update these variables:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";        // Your WiFi name
   const char* password = "YOUR_WIFI_PASSWORD"; // Your WiFi password
   const char* serverIP = "192.168.1.100";     // Your PC's IP address
   ```

## 3. Find Your PC IP Address
- Windows: Open Command Prompt and type `ipconfig`
- Look for "IPv4 Address" (usually 192.168.x.x)
- Use this IP in the ESP32 code

## 4. Upload ESP32 Code
1. Connect ESP32 to computer
2. Select correct board in Arduino IDE (AI Thinker ESP32-CAM)
3. Upload the code
4. Open Serial Monitor to see connection status

## 5. Run Python Server
1. Install required Python packages:
   ```bash
   pip install flask opencv-python mediapipe tensorflow-lite-runtime pyttsx3
   ```
2. Run the server:
   ```bash
   python app.py
   ```

## 6. Test Connection
1. Open browser and go to: `http://localhost:5000/test_camera.html`
2. You should see the ESP32 camera feed
3. The main application will show "Waiting for ESP32..." until feed starts

## 7. Troubleshooting
- **No connection**: Check WiFi credentials and PC IP address
- **Camera not working**: Verify camera connections and power
- **Server not receiving**: Check firewall settings on port 5000

## 8. Network Requirements
- ESP32 and PC must be on same WiFi network
- Port 5000 should be open on PC firewall
- Stable WiFi connection recommended

## Features
- ESP32 sends 640x480 JPEG frames at ~10 FPS
- Server receives frames via HTTP POST
- Real-time hand detection works with wireless feed
- All original features maintained (left hand space, delete key, TTS)
