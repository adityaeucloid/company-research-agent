import streamlit as st
import asyncio
import os
import logging
import nest_asyncio
from crawl_overview import main as crawl_overview
from crawl_financials import main as crawl_financials
from extract_overview import main as extract_overview
from extract_financials import main as extract_financials
import json
from dotenv import load_dotenv

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

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
    .json-output {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        border: 1px solid #ffcdd2;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        border: 1px solid #c8e6c9;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        border: 1px solid #bbdefb;
    }
</style>
""", unsafe_allow_html=True)

def load_json_file(file_path):
    """Load and return JSON data from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}", exc_info=True)
        return None

def run_async_pipeline(company_name: str):
    """Run the complete pipeline for company data extraction."""
    try:
        # Create tabs for different sections
        overview_tab, financials_tab = st.tabs(["Company Overview", "Financial Data"])
        
        with st.spinner("Running company research pipeline..."):
            # Create progress containers
            progress_container = st.empty()
            status_container = st.empty()
            
            # Create new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Run crawling tasks in parallel
                progress_container.info("Starting data crawling...")
                status_container.info("Crawling company overview data...")
                overview_task = loop.create_task(crawl_overview(company_name))
                status_container.info("Crawling financial data...")
                financials_task = loop.create_task(crawl_financials(company_name))
                
                # Wait for crawling to complete
                loop.run_until_complete(asyncio.gather(overview_task, financials_task))
                progress_container.success("Data crawling completed!")
                
                # Run extraction tasks in parallel
                progress_container.info("Starting data extraction...")
                status_container.info("Extracting company overview data...")
                overview_extract_task = loop.create_task(extract_overview(company_name))
                status_container.info("Extracting financial data...")
                financials_extract_task = loop.create_task(extract_financials(company_name))
                
                # Wait for extraction to complete
                loop.run_until_complete(asyncio.gather(overview_extract_task, financials_extract_task))
                progress_container.success("Data extraction completed!")
                
            finally:
                # Clean up the event loop
                loop.close()
            
            # Load and display results
            with overview_tab:
                overview_file = f"crawled_content/{company_name}_extracted.json"
                overview_data = load_json_file(overview_file)
                if overview_data:
                    st.markdown("### Company Overview Data")
                    st.json(overview_data)
                else:
                    st.warning("No overview data available.")
            
            with financials_tab:
                financials_file = f"financial_data/{company_name}_extracted_financial_data.json"
                financials_data = load_json_file(financials_file)
                if financials_data:
                    st.markdown("### Financial Data")
                    st.json(financials_data)
                else:
                    st.warning("No financial data available.")
            
            progress_container.success("Pipeline completed successfully!")
            status_container.empty()
            
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}", exc_info=True)
        st.error(f"Error in pipeline: {str(e)}")

# Main UI
st.title("🏢 Company Research Agent")
st.write("Enter a company name to extract overview and financial data.")

# Input form
with st.form("company_form"):
    company_name = st.text_input(
        "Company Name",
        value="VALUE POINT SYSTEMS PRIVATE LIMITED",
        help="Enter the full company name as registered"
    )
    submitted = st.form_submit_button("Run Research")

# Check API keys
if not os.getenv("TAVILY_API_KEY") or not os.getenv("GEMINI_API_KEY"):
    st.error("""
    Please ensure the following environment variables are set in your .env file:
    - TAVILY_API_KEY
    - GEMINI_API_KEY
    """)
else:
    if submitted:
        if not company_name:
            st.error("Please enter a company name.")
        else:
            run_async_pipeline(company_name)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Company Research Agent | Powered by Streamlit</p>
</div>
""", unsafe_allow_html=True)