# helpers/api_keys.py

import json
import os
import streamlit as st

def load_api_keys():
    """Load API keys from a JSON file if it exists."""
    if os.path.exists("api_keys.json"):
        with open("api_keys.json", 'r') as file:
            data = json.load(file)
            st.session_state.api_keys.update(data)

def save_api_keys():
    """Save API keys to a JSON file."""
    with open("api_keys.json", 'w') as file:
        json.dump(st.session_state.api_keys, file)
