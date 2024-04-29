import streamlit as st
from streamlit_option_menu import option_menu

import streamlit as st
from PIL import Image 
from streamlit import config
from streamlit_option_menu import option_menu
import streamlit as st
from tkinter.tix import COLUMN
from pyparsing import empty

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
from networks import *
import remove_flare
import utils

# 1. as sidebar menu
with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Task'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=1)


# 2. horizontal menu
selected2 = option_menu(None, ["flare removal", "segmentation","depth estimation"], 
    icons=['bi bi-sun', 'bi bi-search', "bi bi-rulers"], 
    menu_icon="cast", default_index=0, orientation="horizontal")



    
    
    
    
    
