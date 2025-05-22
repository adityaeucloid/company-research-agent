import streamlit as st
import sys
import os
import json
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

# Import your existing modules
from extract_overview import extract_company_data
from extract_financials import extract_financial_metrics
from crawl_overview import crawl_company_data
from crawl_financials import crawl_financial_data

# Set page config
st.set_page_config(
    page_title="Company Research Agent",
    page_icon="🏢",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🏢 Company Research Agent")

# Sidebar
st.sidebar.header("Configuration")
company_name = st.sidebar.text_input("Company Name", "VALUE POINT SYSTEMS PRIVATE LIMITED")
max_results = st.sidebar.slider("Maximum Results", 1, 20, 10)

# Main content
tab1, tab2 = st.tabs(["Company Overview", "Financial Data"])

with tab1:
    st.header("Company Overview")
    
    if st.button("Extract Company Overview", key="overview"):
        with st.spinner("Extracting company overview..."):
            try:
                # Create output directory if it doesn't exist
                os.makedirs("company_data", exist_ok=True)
                
                # Crawl and extract data
                overview_data = crawl_company_data(company_name, max_results)
                if overview_data:
                    # Save to JSON
                    output_file = f'company_data/{company_name}_overview.json'
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(overview_data, f, indent=2)
                    
                    # Display results
                    st.success(f"Successfully extracted overview data for {company_name}")
                    
                    # Display structured data
                    st.markdown("### Extracted Data")
                    st.json(overview_data)
                    
                    # Download button
                    with open(output_file, 'r') as f:
                        st.download_button(
                            label="Download JSON",
                            data=f,
                            file_name=f"{company_name}_overview.json",
                            mime="application/json"
                        )
                else:
                    st.error("No overview data found")
            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab2:
    st.header("Financial Data")
    
    if st.button("Extract Financial Data", key="financials"):
        with st.spinner("Extracting financial data..."):
            try:
                # Create output directory if it doesn't exist
                os.makedirs("financial_data", exist_ok=True)
                
                # Crawl and extract data
                financial_data = crawl_financial_data(company_name, max_results)
                if financial_data:
                    # Save to JSON
                    output_file = f'financial_data/{company_name}_financials.json'
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(financial_data, f, indent=2)
                    
                    # Display results
                    st.success(f"Successfully extracted financial data for {company_name}")
                    
                    # Display structured data
                    st.markdown("### Extracted Data")
                    st.json(financial_data)
                    
                    # Download button
                    with open(output_file, 'r') as f:
                        st.download_button(
                            label="Download JSON",
                            data=f,
                            file_name=f"{company_name}_financials.json",
                            mime="application/json"
                        )
                else:
                    st.error("No financial data found")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Company Research Agent") 