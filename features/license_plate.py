# #license_plate_ocr.py
# import streamlit as st
# from models.yolo_inference import run_inference_on_frame
# import cv2
# import tempfile
# st.markdown("# License Plate Detection")
# st.sidebar.markdown("# License Plate")
# # Upload video
# video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

# if video_file is not None:
#     st.video(video_file)

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
#         temp_video.write(video_file.read())
#         temp_video_path = temp_video.name

#     if st.button("Run YOLO Inference"):
#         stframe = st.empty()
#         cap = cv2.VideoCapture(temp_video_path)

#         st.spinner("Running inference... (streaming frame-by-frame)")

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Run inference on the current frame
#             result_frame = run_inference_on_frame(frame, task='license_plate_model')  # Your custom frame-level function

#             # Convert BGR (OpenCV) to RGB (Streamlit)
#             result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

#             # Display the frame
#             stframe.image(result_frame, channels="RGB", use_container_width=True)

#         cap.release()
#         st.success("Inference complete!")

import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
import easyocr  # For OCR
import time

st.markdown("# License Plate Detection")
st.sidebar.markdown("# License Plate")

# Upload video
video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if video_file is not None:
    st.video(video_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        temp_video_path = temp_video.name

    if st.button("Run YOLO Inference"):
        stframe = st.empty()
        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1 / fps if fps > 0 else 0.03

        # Load YOLO model (use your custom model if available)
        model = YOLO("yolov8s.pt")  # Replace with custom plate detection model if needed

        # Initialize EasyOCR
        reader = easyocr.Reader(['en'])

        st.info("Detecting license plates...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)[0]

            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                # Only process if it's a license plate (replace with appropriate label if needed)
                if label.lower() in ['license plate', 'plate']:  # Customize label if needed
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    roi = frame[y1:y2, x1:x2]

                    # Run OCR
                    if roi.size > 0:
                        ocr_result = reader.readtext(roi)
                        plate_text = ocr_result[0][1] if ocr_result else "N/A"
                    else:
                        plate_text = "N/A"

                    # Draw bounding box and OCR result
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, plate_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Convert to RGB and display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_container_width=True)
            time.sleep(delay)

        cap.release()
        st.success("License Plate Detection Complete!")
