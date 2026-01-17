"""
Streamlit Face Recognition App
Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import face_recognition
import pickle
import numpy as np
import os
import time
from datetime import datetime
import pandas as pd
from PIL import Image
import tempfile

# Page config
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1.5rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF9800;
        margin: 1rem 0;
    }
    .face-box {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 10px;
        margin: 10px;
    }
    .stButton button {
        width: 100%;
        background-color: #2196F3;
        color: white;
        font-weight: bold;
    }
    .fps-display {
        background-color: #000;
        color: #0F0;
        padding: 5px 10px;
        border-radius: 5px;
        font-family: monospace;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class FaceRecognitionSystem:
    def __init__(self):
        self.encodings_file = "face_encodings.pkl"
        self.attendance_file = "attendance.csv"
        self.data = self.load_data()
    
    def load_data(self):
        """Load existing face encodings"""
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {"encodings": [], "names": [], "timestamps": []}
        return {"encodings": [], "names": [], "timestamps": []}
    
    def save_data(self):
        """Save face encodings to file"""
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(self.data, f)
    
    def save_attendance(self, name):
        """Save attendance record"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = {"Name": name, "Timestamp": timestamp, "Date": timestamp.split()[0]}
        
        if os.path.exists(self.attendance_file):
            df = pd.read_csv(self.attendance_file)
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        else:
            df = pd.DataFrame([record])
        
        df.to_csv(self.attendance_file, index=False)
        return timestamp
    
    def process_image(self, image):
        """Process image for face detection"""
        # Convert to RGB (face_recognition expects RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        return face_locations, face_encodings
    
    def recognize_faces(self, image):
        """Recognize faces in image"""
        face_locations, face_encodings = self.process_image(image)
        
        recognized_faces = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare with known faces
            if self.data["encodings"]:
                distances = face_recognition.face_distance(self.data["encodings"], face_encoding)
                if len(distances) > 0:
                    min_distance = np.min(distances)
                    if min_distance < 0.6:  # Threshold
                        match_index = np.argmin(distances)
                        name = self.data["names"][match_index]
                        confidence = 1 - min_distance
                    else:
                        name = "Unknown"
                        confidence = 0
                else:
                    name = "Unknown"
                    confidence = 0
            else:
                name = "Unknown (No database)"
                confidence = 0
            
            recognized_faces.append({
                "name": name,
                "location": face_location,
                "confidence": confidence
            })
        
        return recognized_faces
    
    def add_new_face(self, name, image, num_samples=3):
        """Add new face to database"""
        face_locations, face_encodings = self.process_image(image)
        
        if not face_encodings:
            return False, "No face detected in the image"
        
        # Use the first face found
        face_encoding = face_encodings[0]
        
        # Check if name already exists
        if name in self.data["names"]:
            return False, f"Name '{name}' already exists in database"
        
        # Add to database
        self.data["encodings"].append(face_encoding)
        self.data["names"].append(name)
        self.data["timestamps"].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Save to file
        self.save_data()
        
        return True, f"Successfully added '{name}' to database"

def main():
    # Initialize system
    if 'face_system' not in st.session_state:
        st.session_state.face_system = FaceRecognitionSystem()
    
    system = st.session_state.face_system
    
    # Header
    st.markdown("<h1 class='main-header'>👤 Face Recognition System</h1>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/face-id.png", width=100)
        st.markdown("### Navigation")
        
        menu = st.radio(
            "Select Mode:",
            ["🏠 Dashboard", "➕ Add New Face", "📷 Live Recognition", "📊 Attendance Records", "🗄️ Database Management"]
        )
        
        st.markdown("---")
        st.markdown("### System Info")
        st.info(f"👥 Registered: {len(set(system.data['names']))} people")
        st.info(f"📁 Total Samples: {len(system.data['names'])}")
        
        if st.button("🔄 Refresh System"):
            st.rerun()
    
    # Dashboard
    if menu == "🏠 Dashboard":
        st.markdown("<h2 class='sub-header'>Dashboard</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Registered People", len(set(system.data["names"])))
        
        with col2:
            st.metric("Total Face Samples", len(system.data["names"]))
        
        with col3:
            if os.path.exists(system.attendance_file):
                df = pd.read_csv(system.attendance_file)
                st.metric("Attendance Records", len(df))
            else:
                st.metric("Attendance Records", 0)
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📸 Start Live Recognition", use_container_width=True):
                st.session_state.mode = "live"
                st.rerun()
        
        with col2:
            if st.button("➕ Add New Person", use_container_width=True):
                st.session_state.mode = "add"
                st.rerun()
        
        with col3:
            if st.button("📊 View Attendance", use_container_width=True):
                st.session_state.mode = "attendance"
                st.rerun()
        
        # Recent Activity
        st.markdown("### Recent Activity")
        if system.data["timestamps"]:
            recent_data = list(zip(system.data["names"], system.data["timestamps"]))[-5:]
            for name, timestamp in reversed(recent_data):
                st.write(f"✅ **{name}** - {timestamp}")
        else:
            st.warning("No recent activity")
    
    # Add New Face
    elif menu == "➕ Add New Face":
        st.markdown("<h2 class='sub-header'>Add New Face</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            name = st.text_input("Enter person's name:")
            
            # Capture options
            capture_option = st.radio(
                "Capture Method:",
                ["📱 Use Webcam", "📁 Upload Image"]
            )
            
            if capture_option == "📱 Use Webcam":
                picture = st.camera_input("Take a photo")
                if picture:
                    image = Image.open(picture)
                    image_np = np.array(image)
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    
                    # Display captured image
                    st.image(picture, caption="Captured Image", use_column_width=True)
                    
                    if name and st.button("✅ Add to Database", type="primary"):
                        with st.spinner("Processing..."):
                            success, message = system.add_new_face(name, image_bgr)
                            if success:
                                st.success(message)
                                st.balloons()
                            else:
                                st.error(message)
            
            else:  # Upload Image
                uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    image_np = np.array(image)
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    
                    # Display uploaded image
                    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                    
                    if name and st.button("✅ Add to Database", type="primary"):
                        with st.spinner("Processing..."):
                            success, message = system.add_new_face(name, image_bgr)
                            if success:
                                st.success(message)
                                st.balloons()
                            else:
                                st.error(message)
        
        with col2:
            st.markdown("### Tips for Best Results")
            st.markdown("""
            <div class="warning-box">
            <b>📸 Capture Guidelines:</b>
            <ul>
                <li>Ensure good lighting (face should be clearly visible)</li>
                <li>Look directly at the camera</li>
                <li>Remove sunglasses/hats</li>
                <li>Keep a neutral expression</li>
                <li>Make sure face is centered</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Preview database
            if system.data["names"]:
                st.markdown("### Current Database")
                unique_names = sorted(set(system.data["names"]))
                for i, name in enumerate(unique_names, 1):
                    count = system.data["names"].count(name)
                    st.write(f"{i}. **{name}** ({count} sample{'s' if count > 1 else ''})")
    
    # Live Recognition
    elif menu == "📷 Live Recognition":
        st.markdown("<h2 class='sub-header'>Live Face Recognition</h2>", unsafe_allow_html=True)
        
        # Control panel
        col1, col2, col3 = st.columns(3)
        with col1:
            record_attendance = st.checkbox("📝 Record Attendance", value=True)
        with col2:
            show_confidence = st.checkbox("🔢 Show Confidence", value=True)
        with col3:
            detection_model = st.selectbox("Detection Model", ["HOG (Faster)", "CNN (More Accurate)"])
        
        # Start camera
        if st.button("🎥 Start Live Camera", type="primary", use_container_width=True):
            st.session_state.start_camera = True
        
        if st.session_state.get('start_camera', False):
            # Initialize camera
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                st.error("❌ Cannot access webcam. Please check your camera connection.")
                st.stop()
            
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Initialize variables
            if 'fps_data' not in st.session_state:
                st.session_state.fps_data = {
                    'frame_count': 0,
                    'start_time': time.time(),
                    'fps': 0,
                    'last_faces': [],
                    'last_recognition': 0
                }
            
            fps_data = st.session_state.fps_data
            
            # Create placeholders
            frame_placeholder = st.empty()
            results_placeholder = st.empty()
            stats_placeholder = st.empty()
            
            # Create stop button
            stop_col1, stop_col2, stop_col3 = st.columns([1, 2, 1])
            with stop_col2:
                stop_button = st.button("🛑 Stop Camera", use_container_width=True)
            
            while st.session_state.get('start_camera', False) and not stop_button:
                ret, frame = camera.read()
                if not ret:
                    st.error("Failed to grab frame from camera")
                    time.sleep(0.1)
                    continue
                
                current_time = time.time()
                fps_data['frame_count'] += 1
                
                # Calculate FPS every second
                elapsed_time = current_time - fps_data['start_time']
                if elapsed_time > 1.0:
                    fps_data['fps'] = fps_data['frame_count'] / elapsed_time
                    fps_data['frame_count'] = 0
                    fps_data['start_time'] = current_time
                
                # Process frame for recognition (every 0.5 seconds)
                if current_time - fps_data['last_recognition'] > 0.5:
                    # Recognize faces
                    recognized_faces = system.recognize_faces(frame)
                    fps_data['last_faces'] = recognized_faces
                    fps_data['last_recognition'] = current_time
                    
                    # Record attendance
                    if record_attendance:
                        for face in recognized_faces:
                            if face['name'] not in ['Unknown', 'Unknown (No database)']:
                                # Check cooldown for this person
                                person_key = f"last_att_{face['name']}"
                                if current_time - st.session_state.get(person_key, 0) > 10:  # 10 sec cooldown
                                    timestamp = system.save_attendance(face['name'])
                                    st.session_state[person_key] = current_time
                                    st.toast(f"✅ {face['name']} - {timestamp}", icon="✅")
                
                # Draw bounding boxes on the frame
                display_frame = frame.copy()
                
                for face in fps_data['last_faces']:
                    top, right, bottom, left = face['location']
                    name = face['name']
                    confidence = face['confidence']
                    
                    # Choose color based on recognition
                    if 'Unknown' in name:
                        color = (0, 0, 255)  # Red for unknown
                    else:
                        color = (0, 255, 0)   # Green for known
                    
                    # Draw rectangle
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    
                    # Draw label
                    label = name
                    if show_confidence and confidence > 0:
                        label += f" ({confidence:.2f})"
                    
                    y = top - 10 if top - 10 > 10 else top + 10
                    cv2.putText(display_frame, label, (left, y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Display FPS (safe calculation)
                if fps_data['fps'] > 0:
                    fps_text = f"FPS: {fps_data['fps']:.1f}"
                    cv2.putText(display_frame, fps_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Display face count
                face_count = len(fps_data['last_faces'])
                cv2.putText(display_frame, f"Faces: {face_count}", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Convert to RGB for Streamlit
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True, caption="Live Camera Feed")
                
                # Display results
                if fps_data['last_faces']:
                    results_text = f"### 👥 Detected Faces: {len(fps_data['last_faces'])}\n\n"
                    for face in fps_data['last_faces']:
                        if face['confidence'] > 0:
                            results_text += f"**{face['name']}** (Confidence: {face['confidence']:.2%})\n"
                        else:
                            results_text += f"**{face['name']}**\n"
                    results_placeholder.markdown(results_text)
                else:
                    results_placeholder.markdown("### 👤 No faces detected")
                
                # Display statistics
                stats_text = f"""
                ### 📊 Live Statistics
                - **FPS**: {fps_data['fps']:.1f}
                - **Frame Width**: {frame.shape[1]}
                - **Frame Height**: {frame.shape[0]}
                - **Processing Delay**: {current_time - fps_data['last_recognition']:.2f}s
                """
                stats_placeholder.markdown(stats_text)
                
                # Check for stop
                if stop_button:
                    st.session_state.start_camera = False
                    break
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.03)  # ~30 FPS max
            
            # Cleanup
            camera.release()
            cv2.destroyAllWindows()
            
            if stop_button:
                st.success("Camera stopped successfully")
                st.session_state.start_camera = False
                st.rerun()
        else:
            st.info("Click 'Start Live Camera' to begin face recognition")
            
            # Instructions
            st.markdown("""
            ### 📋 Instructions for Live Recognition:
            1. Click **🎥 Start Live Camera** button above
            2. Position yourself in front of the camera
            3. Make sure your face is well-lit
            4. System will automatically detect and recognize faces
            5. Known faces will be marked in **green**
            6. Unknown faces will be marked in **red**
            7. Click **🛑 Stop Camera** when finished
            """)
    
    # Attendance Records
    elif menu == "📊 Attendance Records":
        st.markdown("<h2 class='sub-header'>Attendance Records</h2>", unsafe_allow_html=True)
        
        if os.path.exists(system.attendance_file):
            df = pd.read_csv(system.attendance_file)
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                selected_date = st.date_input("Filter by Date")
            with col2:
                selected_name = st.selectbox("Filter by Name", ["All"] + sorted(df['Name'].unique()))
            
            # Apply filters
            filtered_df = df.copy()
            if selected_date:
                filtered_df = filtered_df[filtered_df['Date'] == str(selected_date)]
            if selected_name != "All":
                filtered_df = filtered_df[filtered_df['Name'] == selected_name]
            
            # Display stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(filtered_df))
            with col2:
                st.metric("Unique People", len(filtered_df['Name'].unique()))
            with col3:
                if not filtered_df.empty:
                    first_record = filtered_df['Timestamp'].min()
                    last_record = filtered_df['Timestamp'].max()
                    st.metric("Time Range", f"{first_record} to {last_record}")
            
            # Show data
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Charts
            if not filtered_df.empty:
                tab1, tab2 = st.tabs(["📈 Daily Count", "👥 Person Stats"])
                
                with tab1:
                    daily_counts = filtered_df.groupby('Date').size().reset_index(name='Count')
                    st.bar_chart(daily_counts.set_index('Date'))
                
                with tab2:
                    person_counts = filtered_df.groupby('Name').size().reset_index(name='Count')
                    st.bar_chart(person_counts.set_index('Name'))
        else:
            st.warning("No attendance records yet")
    
    # Database Management
    elif menu == "🗄️ Database Management":
        st.markdown("<h2 class='sub-header'>Database Management</h2>", unsafe_allow_html=True)
        
        if not system.data["names"]:
            st.warning("Database is empty")
        else:
            # Display current database
            st.markdown("### Current Database")
            
            unique_names = sorted(set(system.data["names"]))
            name_counts = {name: system.data["names"].count(name) for name in unique_names}
            
            for name, count in name_counts.items():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{name}**")
                with col2:
                    st.write(f"{count} sample{'s' if count > 1 else ''}")
                with col3:
                    if st.button(f"Delete", key=f"del_{name}"):
                        # Remove all entries for this name
                        indices_to_remove = [i for i, n in enumerate(system.data["names"]) if n == name]
                        for idx in sorted(indices_to_remove, reverse=True):
                            del system.data["encodings"][idx]
                            del system.data["names"][idx]
                            del system.data["timestamps"][idx]
                        
                        system.save_data()
                        st.success(f"Removed {name} from database")
                        st.rerun()
            
            st.markdown(f"**Total: {len(system.data['names'])} samples for {len(unique_names)} people**")
            
            # Database actions
            st.markdown("---")
            st.markdown("### Database Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Refresh Database", use_container_width=True):
                    system.data = system.load_data()
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Clear All Data", use_container_width=True):
                    if st.checkbox("I understand this will delete ALL data"):
                        system.data = {"encodings": [], "names": [], "timestamps": []}
                        system.save_data()
                        st.success("Database cleared!")
                        st.rerun()
            
            # Export/Import
            st.markdown("### Backup & Restore")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export
                if st.button("📤 Export Database", use_container_width=True):
                    with open(system.encodings_file, "rb") as f:
                        data = f.read()
                    
                    st.download_button(
                        label="Download Backup",
                        data=data,
                        file_name=f"face_database_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                        mime="application/octet-stream"
                    )
            
            with col2:
                # Import
                uploaded_backup = st.file_uploader("Restore from backup", type=['pkl'])
                if uploaded_backup is not None:
                    if st.button("📥 Restore Database", use_container_width=True):
                        try:
                            backup_data = pickle.load(uploaded_backup)
                            system.data = backup_data
                            system.save_data()
                            st.success("Database restored successfully!")
                            st.rerun()
                        except:
                            st.error("Invalid backup file")

if __name__ == "__main__":
    # Initialize session state
    if 'start_camera' not in st.session_state:
        st.session_state.start_camera = False
    
    main()