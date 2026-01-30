# 🚀 Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure ESP32
1. Open `esp32_camera.ino` in Arduino IDE
2. Change these lines:
   ```cpp
   const char* ssid = "YOUR_WIFI_NAME";
   const char* password = "YOUR_WIFI_PASSWORD";
   const char* serverIP = "YOUR_PC_IP";  // Find with `ipconfig`
   ```
3. Upload to ESP32-CAM

### 3. Run the System
```bash
python app.py
```

### 4. Test It
Open browser: `http://localhost:5000/test_camera.html`

## 🎮 Quick Demo

1. **Right Hand**: Make letter signs (A-Z)
2. **Left Hand**: Show to add space
3. **Delete Key**: Remove last character  
4. **Spacebar**: Speak text
5. **ESC**: Exit

## 🔧 Troubleshooting

**No camera feed?**
- Check ESP32 is connected to WiFi
- Verify PC IP address in ESP32 code
- Check firewall on port 5000

**Hand not detected?**
- Ensure good lighting
- Keep hand in camera view
- Check camera focus

**Text not speaking?**
- Check audio output device
- Verify pyttsx3 installation

## 📱 Demo Checklist

Before presenting:
- [ ] ESP32 powered and connected
- [ ] WiFi working (check Serial Monitor)
- [ ] Python server running
- [ ] Camera feed visible in browser
- [ ] Hand gestures recognized
- [ ] Audio output working
- [ ] All controls tested

## 🎯 Demo Flow

1. **Start**: Show wireless camera feed
2. **Basic**: Spell "HELLO" with right hand
3. **Advanced**: Add space with left hand, spell "WORLD"
4. **Correction**: Delete mistake, fix it
5. **Audio**: Press spacebar to speak text
6. **Q&A**: Answer questions about the tech

**Total demo time: 5-7 minutes**
