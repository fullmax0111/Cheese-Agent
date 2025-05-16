import streamlit as st
import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the Streamlit app module
from src.app import *

# The app will run automatically when this file is executed with streamlit run 