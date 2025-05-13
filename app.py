import os
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

# Patch to avoid torch.classes introspection issue
import sys
import types

# Patch torch.classes to avoid crash during Streamlit reload
import torch
torch_classes = types.ModuleType("classes")
torch_classes.__path__ = []  # Prevent Streamlit from inspecting this
torch.classes = torch_classes
sys.modules["torch.classes"] = torch_classes

import streamlit as st
st.title(":rainbow[Smart Traffic Management System]")
st.markdown(
    """ 
    **Play with :rainbow[YOLO]**
    """
)

main_page=st.Page("main.py",title="Welcome to our Home Page of Traffic Management System",icon="🎈")
page_1=st.Page("features/vehicles_detection.py",title="Vehicle Detection",icon = "❄️")
page_2=st.Page("features/crosswalk_pedestrians.py",title="Crosswalk Pedestrians Detection",icon = "❄️")
page_3=st.Page("features/signal_color.py",title="Signal Color Detection",icon = "❄️")
page_4=st.Page("features/license_plate.py",title="License Plate Detection",icon = "❄️")
pg=st.navigation([main_page,page_1,page_2,page_3,page_4])
#pg=st.navigation([main_page])
pg.run()
