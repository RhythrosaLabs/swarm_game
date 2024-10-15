import streamlit as st
import openai
import replicate
import requests
from PIL import Image
from io import BytesIO

# ============================================
#               CONFIGURATION
# ============================================
st.set_page_config(page_title="Enhanced Business Plan Creator", layout="wide")

# Custom CSS to enhance UI
def load_custom_css():
    custom_css = """
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .sidebar .sidebar-content {background-color: #2E8B57; color: white;}
    h1 {color: #4CAF50;}
    """
    st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)

load_custom_css()

# ============================================
#              HELPER FUNCTIONS
# ============================================
def set_api_keys():
    if 'openai_api_key' not in st.session_state or 'replicate_api_key' not in st.session_state:
        st.error("API keys are not set. Please enter them in the sidebar.")
    else:
        openai.api_key = st.session_state['openai_api_key']

def generate_text(prompt, section):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant for generating a {section}."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

def generate_image(prompt):
    try:
        client = replicate.Client(api_token=st.session_state['replicate_api_key'])
        output = client.run(
            "stability-ai/stable-diffusion",
            input={"prompt": prompt}
        )
        return output[0] if output else "Error: No output"
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================
#               MAIN APP
# ============================================
def main():
    with st.sidebar:
        st.header("API Keys")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        replicate_api_key = st.text_input("Replicate API Key", type="password")
        if st.button("Submit API Keys"):
            if openai_api_key and replicate_api_key:
                st.session_state['openai_api_key'] = openai_api_key
                st.session_state['replicate_api_key'] = replicate_api_key
                st.success("API keys saved for the session.")

    if 'openai_api_key' in st.session_state and 'replicate_api_key' in st.session_state:
        set_api_keys()
    else:
        st.warning("Enter your API keys in the sidebar to proceed.")
        return

    st.title("Business Plan Generator")
    st.header("Executive Summary")
    prompt = st.text_area("Enter details for the Executive Summary")
    if st.button("Generate Summary"):
        if prompt:
            with st.spinner("Generating..."):
                summary = generate_text(prompt, "Executive Summary")
            st.write(summary)
        else:
            st.error("Please enter a prompt.")

    st.header("Generate a Business Logo")
    logo_prompt = st.text_input("Enter a prompt for your business logo")
    if st.button("Generate Logo"):
        if logo_prompt:
            with st.spinner("Generating logo..."):
                logo_url = generate_image(logo_prompt)
                if logo_url and not logo_url.startswith("Error"):
                    response = requests.get(logo_url)
                    image = Image.open(BytesIO(response.content))
                    st.image(image, caption="Generated Logo", use_column_width=True)
                else:
                    st.error(logo_url)
        else:
            st.error("Please enter a prompt for the logo.")

if __name__ == "__main__":
    main()
