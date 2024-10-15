import streamlit as st
import openai
import replicate
import requests
from PIL import Image
from io import BytesIO
import zipfile
import re

# Configure Streamlit page
st.set_page_config(page_title="Business Plan Automation", layout="wide")

# Title
st.markdown('<h1 style="text-align: center; color: #2E8B57;">Business Plan Automation</h1>', unsafe_allow_html=True)

# Sidebar for API Keys
with st.sidebar:
    st.header("🔑 API Keys")
    
    st.markdown("**Enter your API keys below.**")
    
    openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    replicate_api_key = st.text_input("Replicate API Key", type="password", placeholder="r8_...")
    
    if st.button("Submit API Keys"):
        if openai_api_key and replicate_api_key:
            st.session_state['openai_api_key'] = openai_api_key
            st.session_state['replicate_api_key'] = replicate_api_key
            st.success("API Keys submitted successfully!")
        else:
            st.error("Please enter both OpenAI and Replicate API keys.")

# Function to set API keys
def set_api_keys():
    if 'openai_api_key' in st.session_state:
        openai.api_key = st.session_state['openai_api_key']
    else:
        st.error("Please enter your OpenAI API Key in the sidebar.")
    
    if 'replicate_api_key' in st.session_state:
        replicate_client = replicate.Client(api_token=st.session_state['replicate_api_key'])
    else:
        st.error("Please enter your Replicate API Key in the sidebar.")

# Function to generate text using OpenAI GPT-4
def generate_text(prompt, section):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are a professional business consultant. Help the user create a detailed and comprehensive {section} for their business plan."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
            n=1,
            stop=None,
        )
        text = response['choices'][0]['message']['content'].strip()
        return text
    except Exception as e:
        return f"Error generating text for {section}: {str(e)}"

# Function to generate images using Replicate
def generate_image(prompt, model_choice):
    try:
        client = replicate.Client(api_token=st.session_state['replicate_api_key'])
        if model_choice == 'Logo Generation':
            output = client.run(
                "stability-ai/stable-diffusion",
                input={"prompt": prompt, "num_outputs":1}
            )
            return output[0] if output else None
        elif model_choice == 'Chart Generation':
            # Placeholder for chart generation logic
            # Replace with an appropriate Replicate model if available
            return None
        else:
            return None
    except Exception as e:
        return f"Error generating image: {str(e)}"

# Function to display image from URL
def display_image_from_url(url, caption):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        st.image(image, caption=caption, use_column_width=True)
    except Exception as e:
        st.error(f"Failed to load image for {caption}: {str(e)}")

# Function to clean generated code (if any)
def clean_text(text):
    text = text.strip()
    text = re.sub(r'^```\w*\n|```$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^.*?Here\'s.*?:\n', '', text, flags=re.DOTALL)
    return text

# Main Content
if 'openai_api_key' not in st.session_state or 'replicate_api_key' not in st.session_state:
    st.warning("Please enter your OpenAI and Replicate API keys in the sidebar to proceed.")
else:
    set_api_keys()
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Executive Summary", 
        "Market Analysis", 
        "Organization & Management", 
        "Service/Product Line", 
        "Marketing & Sales", 
        "Financial Projections"
    ])
    
    # Dictionary to store generated sections
    business_plan = {}
    images = {}
    
    with tab1:
        st.markdown('<h2 style="color: #4682B4;">📄 Executive Summary</h2>', unsafe_allow_html=True)
        prompt = st.text_area("Describe your business for the Executive Summary:", height=150)
        if st.button("Generate Executive Summary", key="exec_summary"):
            if prompt.strip():
                with st.spinner('Generating Executive Summary...'):
                    summary = generate_text(prompt, "Executive Summary")
                    business_plan['Executive Summary'] = summary
                st.success("Executive Summary generated successfully!")
                st.write(summary)
            else:
                st.error("Please provide a description of your business.")
    
    with tab2:
        st.markdown('<h2 style="color: #4682B4;">📈 Market Analysis</h2>', unsafe_allow_html=True)
        prompt = st.text_area("Provide details about your target market and competition:", height=150)
        if st.button("Generate Market Analysis", key="market_analysis"):
            if prompt.strip():
                with st.spinner('Generating Market Analysis...'):
                    analysis = generate_text(prompt, "Market Analysis")
                    business_plan['Market Analysis'] = analysis
                st.success("Market Analysis generated successfully!")
                st.write(analysis)
            else:
                st.error("Please provide details about your target market and competition.")
    
    with tab3:
        st.markdown('<h2 style="color: #4682B4;">👥 Organization & Management</h2>', unsafe_allow_html=True)
        prompt = st.text_area("Describe your business's organizational structure and management team:", height=150)
        if st.button("Generate Organization & Management", key="org_management"):
            if prompt.strip():
                with st.spinner('Generating Organization & Management...'):
                    org_management = generate_text(prompt, "Organization & Management")
                    business_plan['Organization & Management'] = org_management
                st.success("Organization & Management generated successfully!")
                st.write(org_management)
            else:
                st.error("Please describe your business's organizational structure and management team.")
    
    with tab4:
        st.markdown('<h2 style="color: #4682B4;">🛠️ Service/Product Line</h2>', unsafe_allow_html=True)
        prompt = st.text_area("Describe your products or services:", height=150)
        if st.button("Generate Service/Product Line", key="service_product"):
            if prompt.strip():
                with st.spinner('Generating Service/Product Line...'):
                    service_product = generate_text(prompt, "Service/Product Line")
                    business_plan['Service/Product Line'] = service_product
                st.success("Service/Product Line generated successfully!")
                st.write(service_product)
            else:
                st.error("Please describe your products or services.")
        
        st.markdown('<h3 style="color: #4682B4;">🎨 Generate Business Logo</h3>', unsafe_allow_html=True)
        logo_prompt = st.text_input("Enter a prompt for your business logo:", "Modern minimalist logo for a tech startup")
        logo_model = st.selectbox("Select Image Generation Model:", ["Logo Generation", "Chart Generation"], index=0)
        if st.button("Generate Logo", key="generate_logo"):
            if logo_prompt.strip():
                with st.spinner('Generating Logo...'):
                    logo_url = generate_image(logo_prompt, logo_model)
                    if logo_url and isinstance(logo_url, str) and not logo_url.startswith("Error"):
                        images['Business Logo'] = logo_url
                        st.success("Logo generated successfully!")
                        display_image_from_url(logo_url, "Business Logo")
                    else:
                        st.error(logo_url)
            else:
                st.error("Please enter a prompt for the logo.")
    
    with tab5:
        st.markdown('<h2 style="color: #4682B4;">📢 Marketing & Sales</h2>', unsafe_allow_html=True)
        prompt = st.text_area("Outline your marketing and sales strategies:", height=150)
        if st.button("Generate Marketing & Sales", key="marketing_sales"):
            if prompt.strip():
                with st.spinner('Generating Marketing & Sales Strategy...'):
                    marketing_sales = generate_text(prompt, "Marketing & Sales")
                    business_plan['Marketing & Sales'] = marketing_sales
                st.success("Marketing & Sales Strategy generated successfully!")
                st.write(marketing_sales)
            else:
                st.error("Please outline your marketing and sales strategies.")
    
    with tab6:
        st.markdown('<h2 style="color: #4682B4;">💰 Financial Projections</h2>', unsafe_allow_html=True)
        prompt = st.text_area("Provide your financial projections and funding requirements:", height=150)
        if st.button("Generate Financial Projections", key="financial_projections"):
            if prompt.strip():
                with st.spinner('Generating Financial Projections...'):
                    financial = generate_text(prompt, "Financial Projections")
                    business_plan['Financial Projections'] = financial
                st.success("Financial Projections generated successfully!")
                st.write(financial)
            else:
                st.error("Please provide your financial projections and funding requirements.")
        
        st.markdown('<h3 style="color: #4682B4;">📊 Generate Financial Chart</h3>', unsafe_allow_html=True)
        chart_prompt = st.text_input("Enter a prompt for your financial chart:", "Bar chart showing projected revenue for the next 5 years")
        chart_model = st.selectbox("Select Image Generation Model:", ["Chart Generation"], index=0)
        if st.button("Generate Chart", key="generate_chart"):
            if chart_prompt.strip():
                with st.spinner('Generating Financial Chart...'):
                    chart_url = generate_image(chart_prompt, chart_model)
                    if chart_url and isinstance(chart_url, str) and not chart_url.startswith("Error"):
                        images['Financial Chart'] = chart_url
                        st.success("Financial Chart generated successfully!")
                        display_image_from_url(chart_url, "Financial Chart")
                    else:
                        st.error(chart_url)
            else:
                st.error("Please enter a prompt for the financial chart.")
    
    # Generate and Download Business Plan
    st.markdown('<h2 style="text-align: center; color: #2E8B57;">📥 Download Your Business Plan</h2>', unsafe_allow_html=True)
    if st.button("Download Business Plan as ZIP"):
        if business_plan or images:
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                # Add text sections
                for section, content in business_plan.items():
                    zip_file.writestr(f"{section}.txt", content)
                
                # Add images
                for img_name, img_url in images.items():
                    if isinstance(img_url, str) and img_url.startswith("http"):
                        try:
                            img_response = requests.get(img_url)
                            img_response.raise_for_status()
                            img = Image.open(BytesIO(img_response.content))
                            img_buffer = BytesIO()
                            img.save(img_buffer, format='PNG')
                            zip_file.writestr(f"{img_name}.png", img_buffer.getvalue())
                        except:
                            pass  # Skip if there's an error
                    else:
                        pass  # Skip if not a valid URL
                
                # Add additional files if necessary
                
            zip_bytes = zip_buffer.getvalue()
            st.download_button(
                label="Download Business Plan ZIP",
                data=zip_bytes,
                file_name="business_plan.zip",
                mime="application/zip",
                help="Download a ZIP file containing all generated sections and images."
            )
        else:
            st.error("No content to download. Please generate sections of your business plan first.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center;">
            <p>Created by [Your Name](https://yourwebsite.com) | 
            [GitHub](https://github.com/yourusername) | 
            [LinkedIn](https://linkedin.com/in/yourusername)</p>
        </div>
        """, unsafe_allow_html=True)
