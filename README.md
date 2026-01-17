# **Face Recognition System** 👤

A complete, user-friendly face recognition system built with Streamlit. This application allows you to capture faces, recognize them in real-time via webcam, and maintain attendance records - all through an intuitive web interface.

## **✨ Features**

### **🎯 Core Features**
- **Live Face Capture**: Add new faces via webcam or image upload
- **Real-Time Recognition**: Instant face recognition using webcam
- **Attendance System**: Automatic attendance recording with timestamps
- **Database Management**: View, edit, and backup face data
- **Multiple Face Support**: Detect and recognize multiple people simultaneously

### **📊 Dashboard Features**
- System overview and statistics
- Quick access to all features
- Recent activity tracking
- Performance metrics

### **🔧 Technical Features**
- **CPU Optimized**: Uses HOG model for efficient CPU processing
- **Real-Time Processing**: Live webcam feed with face detection
- **Confidence Scoring**: Shows recognition confidence for each face
- **Adjustable Settings**: Customize detection parameters
- **Data Export**: Export attendance records as CSV
- **Database Backup**: Backup and restore functionality

## **🚀 Quick Start**

### **1. Installation**

```bash
# Clone the repository
git clone <your-repo-url>
cd Face_recognition-master

# Create virtual environment (optional but recommended)
python -m venv faceenv

# Activate virtual environment
# On Windows:
faceenv\Scripts\activate
# On macOS/Linux:
source faceenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Run the Application**

```bash
streamlit run app.py
```

Then open your browser and navigate to: `http://localhost:8501`

## **📁 File Structure**

```
Face_recognition-master/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── face_encodings.pkl       # Face database (auto-generated)
├── attendance.csv           # Attendance records (auto-generated)
├── faceenv/                 # Virtual environment (optional)
└── README.md                # This file
```

## **🖥️ System Requirements**

### **Minimum Requirements**
- **Processor**: Intel Core i3 or equivalent
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 500MB free space
- **Webcam**: Required for live recognition
- **Python**: 3.8 or higher

### **Recommended (Your System)**
- **Processor**: 12th Gen Intel Core i5-1235U ✅
- **RAM**: 16.0 GB ✅
- **OS**: Windows 10/11 64-bit ✅

## **🔧 How to Use**

### **Step 1: Add Faces to Database**
1. Go to "➕ Add New Face" in the sidebar
2. Enter the person's name
3. Choose capture method:
   - **Webcam**: Take a photo directly
   - **Upload**: Use existing image
4. Click "Add to Database"

### **Step 2: Live Recognition**
1. Go to "📷 Live Recognition"
2. Adjust settings as needed:
   - Enable/disable attendance recording
   - Show/hide confidence scores
3. Click "🎥 Start Live Camera"
4. System will automatically recognize faces
5. Known faces: **Green** bounding boxes
6. Unknown faces: **Red** bounding boxes
7. Click "🛑 Stop Camera" when finished

### **Step 3: View Attendance**
1. Go to "📊 Attendance Records"
2. Filter by date or person
3. View statistics and charts
4. Download records as CSV

### **Step 4: Manage Database**
1. Go to "🗄️ Database Management"
2. View all registered faces
3. Delete individual entries
4. Backup or restore database
5. Clear entire database if needed

## **⚙️ Configuration**

### **Key Settings**
- **Face Confidence Threshold**: 0.6 (adjust in code)
- **Detection Model**: HOG (faster) or CNN (more accurate)
- **Processing Frequency**: Every 0.5 seconds
- **Attendance Cooldown**: 10 seconds per person

### **File Locations**
- **Face Database**: `face_encodings.pkl`
- **Attendance Records**: `attendance.csv`
- **Backup Files**: User-specified location

## **🛠️ Troubleshooting**

### **Common Issues & Solutions**

#### **1. "Cannot access webcam"**
- Check if another application is using the camera
- Grant camera permissions to the browser
- Test camera with another application first

#### **2. "No face detected"**
- Ensure good lighting on the face
- Face should be centered in the frame
- Remove sunglasses/hats that obscure the face

#### **3. "Low recognition accuracy"**
- Add multiple face samples per person
- Capture faces in different lighting conditions
- Ensure clear, high-quality images
- Adjust confidence threshold in code

#### **4. "Performance is slow"**
- Reduce camera resolution in code (line 395-396)
- Increase processing interval (line 416)
- Close other applications using the camera

#### **5. "Module not found" errors**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or install individually
pip install streamlit opencv-python face-recognition numpy pandas pillow
```

### **Error Logs**
Check Streamlit logs for detailed error information:
```bash
streamlit run app.py --logger.level=debug
```

## **📊 Performance Tips**

### **For Optimal Performance:**
1. **Good Lighting**: Ensure face is well-lit
2. **Simple Background**: Use plain background when adding faces
3. **Multiple Samples**: Add 3-5 samples per person
4. **Regular Database Cleanup**: Remove old/unused entries
5. **Close Background Apps**: Free up CPU resources

### **Expected Performance:**
- **Detection Speed**: 15-30 FPS (depends on CPU)
- **Recognition Accuracy**: 85-95% with good samples
- **Memory Usage**: 200-500MB
- **Processing Time**: < 0.5 seconds per frame

## **🔒 Privacy & Security**

### **Data Storage**
- Face encodings are stored locally in `face_encodings.pkl`
- Attendance records are stored locally in `attendance.csv`
- No data is sent to external servers
- Backups are stored in user-specified locations

### **Privacy Features**
- Local processing only
- No cloud data transmission
- User controls all data
- Easy data deletion options

## **📈 Future Enhancements**

### **Planned Features**
- [ ] Face mask detection
- [ ] Emotion recognition
- [ ] Age and gender estimation
- [ ] Cloud synchronization
- [ ] Mobile app version
- [ ] API for integration

### **Advanced Features**
- [ ] Deep learning models
- [ ] 3D face recognition
- [ ] Anti-spoofing measures
- [ ] Batch processing
- [ ] Custom model training

## **🤝 Contributing**

### **How to Contribute**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### **Development Setup**
```bash
# Clone the repository
git clone https://github.com/shushonk/Face_recognition.git

# Create virtual environment
python -m venv venv

# Activate and install dev dependencies
pip install -r requirements.txt
pip install pytest pylint black
```

## **📚 Learning Resources**

### **Related Technologies**
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenCV Face Recognition](https://docs.opencv.org/)
- [Face Recognition Library](https://github.com/ageitgey/face_recognition)
- [Dlib Documentation](http://dlib.net/)

### **Tutorials**
- [Face Recognition with Python](https://realpython.com/face-recognition-with-python/)
- [Streamlit Tutorials](https://docs.streamlit.io/library/get-started)
- [OpenCV Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html)

## **📄 License**

This project is licensed under the MIT License - see the LICENSE file for details.

## **🙏 Acknowledgments**

- Built with [Streamlit](https://streamlit.io/)
- Face recognition powered by [face_recognition](https://github.com/ageitgey/face_recognition)
- Computer vision with [OpenCV](https://opencv.org/)
- Icons by [Icons8](https://icons8.com/)

## **📞 Support**

### **Getting Help**
- Check the [Troubleshooting](#troubleshooting) section
- Review [Common Issues](#common-issues--solutions)
- Search existing GitHub issues

### **Report Issues**
1. Check if issue already exists
2. Provide detailed error message
3. Include system specifications
4. Add steps to reproduce

---

**Happy Face Recognizing!** 🎉👤✨

---

