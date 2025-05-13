#main.py
import streamlit as st
import tempfile
import cv2
import os
from pathlib import Path
from models.yolo_inference import run_inference_on_frame 
import streamlit as st
st.markdown("# Home 🎈")
st.sidebar.markdown("# Home 🎈")

