# import streamlit as st
# import tempfile
# import cv2
# import os
# from pathlib import Path
# import time


# st.markdown("# Vehicles Detection")
# st.sidebar.markdown("# Vehicles Detection")

# video_file = st.file_uploader("Upload a video file for Vehicle Detection", type=["mp4", "mov", "avi"], key="vehicle_detection")

# expected_video_path = Path(__file__).parent.parent / "data" / "expected_output" / "vehicle_detection_expected_video.mp4"
# expected_video_path = str(expected_video_path)

# if video_file is not None:
#     st.video(video_file)

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
#         temp_video.write(video_file.read())
#         temp_video_path = temp_video.name

#     if st.button("Run Vehicle Detection"):
#         stframe = st.empty()
#         cap = cv2.VideoCapture(temp_video_path)


#         if not cap.isOpened():
#             st.error("Could not open uploaded video.")
#         else:
            
#             st.spinner("Running vehicle detection... (streaming frame-by-frame)")
#             fps = cap.get(cv2.CAP_PROP_FPS)
#             delay = 1 / fps if fps > 0 else 0.1  # fallback to 10 FPS if unknown
#             stframe = st.empty()
        
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 # Your detection logic
#                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
#                 contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#                 vehicle_count = 0
#                 for contour in contours:
#                     x, y, w, h = cv2.boundingRect(contour)
#                     if 50 < w < 500 and 30 < h < 400:
#                         vehicle_count += 1
#                         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                         cv2.putText(frame, "Vehicle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#                 cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 stframe.image(frame_rgb, channels="RGB", use_container_width=True)
#                 time.sleep(delay)  # Control playback speed

#             cap.release()
#             st.success("Vehicle detection complete!")

# # Frame-by-Frame Playback for Expected Video
# if os.path.exists(expected_video_path):
#     st.markdown("### Colab Output Video")

#     play = st.checkbox("Play/Pause", value=True)
#     stframe_colab = st.empty()

#     cap_colab = cv2.VideoCapture(expected_video_path)
#     fps = cap_colab.get(cv2.CAP_PROP_FPS)
#     delay = 1 / fps if fps > 0 else 0.03  # Fallback delay

#     frame_idx = 0
#     frame_cache = []
#     total_frames = int(cap_colab.get(cv2.CAP_PROP_FRAME_COUNT))

#     while cap_colab.isOpened() and frame_idx < total_frames:
#         if len(frame_cache) <= frame_idx:
#             ret, frame = cap_colab.read()
#             if not ret:
#                 break
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame_cache.append(frame_rgb)

#         if play:
#             stframe_colab.image(frame_cache[frame_idx], channels="RGB", use_container_width=True)
#             frame_idx += 1
#             time.sleep(delay)
#         else:
#             time.sleep(0.1)

#     cap_colab.release()
# else:
#     st.error("Colab output video not found at the specified path.")

import streamlit as st
import tempfile
import cv2
import os
from pathlib import Path
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8s.pt") 

st.markdown("# Vehicles Detection")
st.sidebar.markdown("# Vehicles Detection")

video_file = st.file_uploader("Upload a video file for Vehicle Detection", type=["mp4", "mov", "avi"], key="vehicle_detection")

expected_video_path = Path(__file__).parent.parent / "data" / "expected_output" / "vehicle_detection_expected_video.mp4"
expected_video_path = str(expected_video_path)

if video_file is not None:
    st.video(video_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        temp_video_path = temp_video.name

    if st.button("Run Vehicle Detection"):
        stframe = st.empty()
        cap = cv2.VideoCapture(temp_video_path)

        if not cap.isOpened():
            st.error("Could not open uploaded video.")
        else:
            st.spinner("Running vehicle detection... (frame-by-frame)")
            fps = cap.get(cv2.CAP_PROP_FPS)
            delay = 1 / fps if fps > 0 else 0.1

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLOv8 detection
                results = model(frame)
                detections = results[0].boxes

                vehicle_count = 0
                for box in detections:
                    cls_id = int(box.cls)
                    label = model.names[cls_id]
                    if label in ["car", "truck", "bus", "motorbike"]:
                        vehicle_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", use_container_width=True)
                time.sleep(delay)

            cap.release()
            st.success("Vehicle detection complete!")

# Frame-by-frame playback of expected video
if os.path.exists(expected_video_path):
    st.markdown("### Colab Output Video")

    play = st.checkbox("Play/Pause", value=True)
    stframe_colab = st.empty()

    cap_colab = cv2.VideoCapture(expected_video_path)
    fps = cap_colab.get(cv2.CAP_PROP_FPS)
    delay = 1 / fps if fps > 0 else 0.03

    frame_idx = 0
    frame_cache = []
    total_frames = int(cap_colab.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap_colab.isOpened() and frame_idx < total_frames:
        if len(frame_cache) <= frame_idx:
            ret, frame = cap_colab.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_cache.append(frame_rgb)

        if play:
            stframe_colab.image(frame_cache[frame_idx], channels="RGB", use_container_width=True)
            frame_idx += 1
            time.sleep(delay)
        else:
            time.sleep(0.1)

    cap_colab.release()
else:
    st.error("Colab output video not found at the specified path.")
