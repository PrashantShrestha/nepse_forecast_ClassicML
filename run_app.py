# run_app.py
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import and run the Streamlit app
from src.app.app import main

if __name__ == "__main__":
    main()