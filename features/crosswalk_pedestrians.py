# import streamlit as st
# import cv2
# import os
# import numpy as np
# from pathlib import Path
# from ultralytics import YOLO
# import tempfile
# import time

# st.markdown("# Crosswalk Pedestrians Detection")
# st.sidebar.markdown("# Pedestrians")

# # Video file uploader
# video_file = st.file_uploader("Upload a video for pedestrian detection", type=["mp4", "mov", "avi"], key="pedestrian_detection")
# expected_video_path = Path(__file__).parent.parent / "data" / "expected_output" / "pedetrians_expected_video.mp4"
# expected_video_path = str(expected_video_path)

# if video_file is not None:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
#         temp_video.write(video_file.read())
#         video_path = temp_video.name

#     # Inference Button
#     if st.button("Run Inference"):
#         model = YOLO('../models/crosswalk_pedestrian_detection.pt')

#         cap = cv2.VideoCapture(video_path)
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#         PEDESTRIAN_ID = 0
#         ZEBRA_CROSSING_ID = 9

#         stframe = st.empty()

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             results = model(frame)[0]

#             for box in results.boxes:
#                 cls_id = int(box.cls[0])
#                 conf = float(box.conf[0])
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])

#                 if cls_id == PEDESTRIAN_ID:
#                     label = f"Person {conf:.2f}"
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#                 elif cls_id == ZEBRA_CROSSING_ID:
#                     label = f"Zebra {conf:.2f}"
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
#                     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

#             # Convert BGR to RGB for Streamlit
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             stframe.image(frame_rgb, channels="RGB", use_container_width=True)

#         cap.release()

# # Display Colab Output Video
# if os.path.exists(expected_video_path):
#     st.markdown("### Colab Output Video")

#     play = st.checkbox("Play/Pause", value=True)
#     stframe_colab = st.empty()

#     cap_colab = cv2.VideoCapture(expected_video_path)
#     fps = cap_colab.get(cv2.CAP_PROP_FPS)
#     delay = 1 / fps if fps > 0 else 0.03

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
import cv2
import os
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import tempfile
import time

st.markdown("# Crosswalk Pedestrians Detection")
st.sidebar.markdown("# Pedestrians")

# Upload video
video_file = st.file_uploader("Upload a video for pedestrian detection", type=["mp4", "mov", "avi"], key="pedestrian_detection")

# Expected video path for Colab result display
expected_video_path = Path(__file__).parent.parent / "data" / "expected_output" / "pedetrians_expected_video.mp4"
expected_video_path = str(expected_video_path)

if video_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    if st.button("Run Inference"):
        stframe = st.empty()
        model = YOLO("yolov8s.pt")  # Using default YOLOv8s model

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1 / fps if fps > 0 else 0.03

        st.info("Detecting pedestrians in video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)[0]  # Inference on single frame

            person_count = 0

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                label = model.names[cls_id]
                if label == "person":
                    person_count += 1
                    display_label = f"{label} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, display_label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(frame, f"Persons Detected: {person_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_container_width=True)
            time.sleep(delay)

        cap.release()
        st.success("Pedestrian detection completed!")

# Display Colab-generated video
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
