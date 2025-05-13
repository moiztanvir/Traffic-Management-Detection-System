# #signal_color.py
# import streamlit as st
# import tempfile
# import cv2
# from models.yolo_inference import run_inference_on_frame
# from pathlib import Path
# import os
# import time
# st.markdown("# Traffic Signal Color Detection")
# st.sidebar.markdown("# Traffic Signal Color")

# # Upload video for Signal Color Detection
# signal_video_file = st.file_uploader("Upload a video for Signal Color Detection", type=["mp4", "mov", "avi"], key="signal")
# expected_video_path = Path(__file__).parent.parent / "data" / "expected_output" / "singal_color_detection_expected_video.mp4"
# expected_video_path = str(expected_video_path)

# if signal_video_file is not None:
#     st.video(signal_video_file)

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
#         temp_video.write(signal_video_file.read())
#         signal_video_path = temp_video.name

#     if st.button("Run YOLO Inference for Signal Color"):
#         stframe = st.empty()
#         cap = cv2.VideoCapture(signal_video_path)

#         st.spinner("Running inference on signal colors... (streaming frame-by-frame)")

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Run inference on the current frame using the signal color detection model
#             result_frame = run_inference_on_frame(frame, task='signal_color_detection')

#             # Convert BGR (OpenCV) to RGB (Streamlit)
#             result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

#             # Display the frame
#             stframe.image(result_frame, channels="RGB", use_container_width=True)

#         cap.release()
#         st.success("Signal Color Detection Complete!")

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
from ultralytics import YOLO
from pathlib import Path
import os
import time
import numpy as np

st.markdown("# Traffic Signal Color Detection")
st.sidebar.markdown("# Traffic Signal Color")

# Upload video
signal_video_file = st.file_uploader("Upload a video for Signal Color Detection", type=["mp4", "mov", "avi"], key="signal")

# Path to expected output
expected_video_path = Path(__file__).parent.parent / "data" / "expected_output" / "singal_color_detection_expected_video.mp4"
expected_video_path = str(expected_video_path)

if signal_video_file is not None:
    st.video(signal_video_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(signal_video_file.read())
        signal_video_path = temp_video.name

    if st.button("Run YOLO Inference for Signal Color"):
        stframe = st.empty()
        cap = cv2.VideoCapture(signal_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1 / fps if fps > 0 else 0.03

        model = YOLO("yolov8s.pt")  # Use appropriate YOLOv8 model
        st.info("Detecting traffic lights and identifying colors...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)[0]

            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                if label == "traffic light":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Extract region of interest (ROI)
                    roi = frame[y1:y2, x1:x2]

                    if roi.size == 0:
                        continue

                    # Convert to HSV and analyze average hue
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    avg_hue = np.mean(hsv[:, :, 0])
                    avg_sat = np.mean(hsv[:, :, 1])
                    avg_val = np.mean(hsv[:, :, 2])

                    # Simple heuristic for signal color classification
                    if avg_val < 50:
                        signal_color = "Off"
                        color = (128, 128, 128)
                    elif avg_hue < 30:
                        signal_color = "Red"
                        color = (0, 0, 255)
                    elif 30 <= avg_hue < 70:
                        signal_color = "Yellow"
                        color = (0, 255, 255)
                    else:
                        signal_color = "Green"
                        color = (0, 255, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{signal_color}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Display updated frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_container_width=True)
            time.sleep(delay)

        cap.release()
        st.success("Signal Color Detection Complete!")

# Display expected output (Colab-generated video)
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
