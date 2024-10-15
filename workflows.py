# ui/tabs/workflows.py

import streamlit as st
from helpers.file_management import add_file_to_global_storage
from helpers.content_generation import generate_file_with_gpt
from utils.zip_utils import create_zip

def custom_workflows_tab():
    """Custom Workflows Tab"""
    st.header("📂 Custom Workflows")
    st.write("Create custom automated workflows.")
    if "workflow_steps" not in st.session_state:
        st.session_state["workflow_steps"] = []

    def add_step():
        st.session_state["workflow_steps"].append({"prompt": "", "file_name": "", "file_data": None})

    if st.button("➕ Add Step", key="add_workflow_step_button"):
        add_step()

    for i, step in enumerate(st.session_state["workflow_steps"]):
        st.write(f"### Step {i + 1}")
        step["prompt"] = st.text_input(f"Prompt for step {i + 1}", value=step["prompt"], key=f"workflow_prompt_{i}")
        if st.button("➖ Remove Step", key=f"remove_workflow_step_{i}"):
            st.session_state["workflow_steps"].pop(i)
            st.experimental_rerun()

    if st.button("Generate All Files", key="generate_all_workflow_files_button"):
        for i, step in enumerate(st.session_state["workflow_steps"]):
            if step["prompt"].strip():
                with st.spinner(f"Generating file for step {i + 1}..."):
                    file_name, file_data = generate_file_with_gpt(step["prompt"])
                    if file_name and file_data:
                        step["file_name"] = file_name
                        step["file_data"] = file_data
                        add_file_to_global_storage(file_name, file_data)
                        st.success(f"File for step {i + 1} generated: {file_name}")
            else:
                st.warning(f"Prompt for step {i + 1} is empty.")

    if st.button("Download Workflow Files as ZIP", key="download_workflow_zip_button"):
        with st.spinner("Creating ZIP file..."):
            zip_buffer = create_zip({step["file_name"]: step["file_data"] for step in st.session_state["workflow_steps"] if step["file_data"]})
            st.download_button(
                label="Download ZIP",
                data=zip_buffer.getvalue(),
                file_name="workflow_files.zip",
                mime="application/zip"
            )

