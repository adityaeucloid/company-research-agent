import streamlit as st
import asyncio
import json
import os
import pandas as pd
from dotenv import load_dotenv
from combined_financials import main as process_financials
from detailed_overview import main as process_overview
import logging
from datetime import datetime
import pathlib

# Create necessary directories
DATA_DIR = pathlib.Path("data")
FINANCIAL_DATA_DIR = DATA_DIR / "financial_data"
LOGS_DIR = DATA_DIR / "logs"

for directory in [DATA_DIR, FINANCIAL_DATA_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate environment variables
required_env_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY", "GEMINI_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    st.stop()

# Streamlit app configuration
st.set_page_config(
    page_title="Company Research Agent",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 16px;
        margin: 10px 0;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .sidebar-title {
        margin-bottom: 0.5rem !important;
    }
    .sidebar-instructions {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    .sidebar-input {
        margin-top: 0.5rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'financial_data' not in st.session_state:
    st.session_state.financial_data = None
if 'overview_data' not in st.session_state:
    st.session_state.overview_data = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

def convert_to_dataframe(data):
    """Convert nested dictionary to flat dataframe."""
    if not data:
        return pd.DataFrame()
    
    # Flatten nested dictionaries
    flat_dict = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat_dict[f"{key} - {subkey}"] = subvalue
        elif isinstance(value, list):
            flat_dict[key] = ", ".join(str(item) for item in value)
        else:
            flat_dict[key] = value
    
    return pd.DataFrame([flat_dict])

def flatten_dict(d, parent_key='', sep=' - '):
    """Flatten nested dictionary with custom separator."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            if all(isinstance(item, dict) for item in v):
                # Handle list of dictionaries
                for i, item in enumerate(v):
                    items.extend(flatten_dict(item, f"{new_key} {i+1}", sep=sep).items())
            else:
                # Handle simple list
                items.append((new_key, ', '.join(str(x) for x in v)))
        else:
            items.append((new_key, v))
    return dict(items)

def extract_expandable_sections(data):
    """Extract sections that should be displayed as expandable tables."""
    expandable_sections = {
        'Current Directors & Key Managerial Personnel': [],
        'Subsidiary Companies': [],
        'GST Details': []
    }
    
    main_data = data.copy()
    
    # Extract expandable sections
    for section in expandable_sections.keys():
        if section in data:
            expandable_sections[section] = data[section]
            del main_data[section]
    
    # Extract GST details from main data
    gst_details = []
    for key, value in list(main_data.items()):
        if key.startswith('GST'):
            if isinstance(value, list):
                gst_details.extend(value)
            else:
                gst_details.append(value)
            del main_data[key]
    
    if gst_details:
        expandable_sections['GST Details'] = gst_details
    
    return main_data, expandable_sections

def organize_financial_data(data):
    """Organize financial data into clear sections."""
    if not data:
        return {}, {}
    
    # Initialize sections
    charge_details = {}
    financial_metrics = {}
    other_data = {}
    
    # Process each field
    for key, value in data.items():
        if key == 'charge_details_in_cr':
            charge_details = value
        elif key == 'financial_metrics_YoY_growth':
            financial_metrics = value
        else:
            other_data[key] = value
    
    return {
        'Charge Details': charge_details,
        'Financial Metrics YoY': financial_metrics,
        'Other Information': other_data
    }

def convert_to_string(value):
    """Convert any value to string format."""
    if isinstance(value, (int, float)):
        return f"{value:,.2f}"
    elif isinstance(value, dict):
        return json.dumps(value, indent=2)
    elif isinstance(value, list):
        return ", ".join(str(item) for item in value)
    else:
        return str(value)

def display_financial_data(data, section_name):
    """Display financial data in a structured format."""
    if not data:
        st.warning(f"No {section_name} data available")
        return

    # Organize data into sections
    sections = {
        "Charge Details": {},
        "Financial Metrics (YoY Growth)": {},
        "Other Information": {}
    }

    # Extract charge details
    if "charge_details_in_cr" in data:
        charge_details = data["charge_details_in_cr"]
        if isinstance(charge_details, dict):
            # Extract main charge metrics
            for key, value in charge_details.items():
                if key != "Charges Breakdown by Lending Institution":
                    sections["Charge Details"][key] = value

            # Handle charges breakdown
            if "Charges Breakdown by Lending Institution" in charge_details:
                breakdown = charge_details["Charges Breakdown by Lending Institution"]
                if isinstance(breakdown, dict):
                    with st.expander("📊 Charges Breakdown by Lending Institution", expanded=True):
                        breakdown_df = pd.DataFrame([
                            {"Lender": k, "Amount": v} for k, v in breakdown.items()
                        ])
                        st.dataframe(
                            breakdown_df,
                            column_config={
                                "Lender": st.column_config.TextColumn(
                                    "Lender",
                                    width="medium"
                                ),
                                "Amount": st.column_config.TextColumn(
                                    "Amount",
                                    width="medium"
                                )
                            },
                            hide_index=True,
                            use_container_width=True
                        )

    # Extract financial metrics
    if "financial_metrics_YoY_growth" in data:
        sections["Financial Metrics (YoY Growth)"] = data["financial_metrics_YoY_growth"]

    # Extract other information
    for key, value in data.items():
        if key not in ["charge_details_in_cr", "financial_metrics_YoY_growth"]:
            sections["Other Information"][key] = value

    # Display each section
    for section_name, section_data in sections.items():
        if section_data:
            st.subheader(section_name)
            display_nested_data(section_data, section_name)

def display_nested_data(data, section_name):
    """Display nested data in a clean table format."""
    if not data:
        st.warning(f"No {section_name} data available")
        return

    # Create a DataFrame from the data
    df = pd.DataFrame([
        {"Field": k, "Value": v} for k, v in data.items()
    ])
    
    # Sort fields alphabetically
    df = df.sort_values("Field")
    
    # Display the table with custom styling
    st.dataframe(
        df,
        column_config={
            "Field": st.column_config.TextColumn(
                "Field",
                width="medium",
                help="Field name"
            ),
            "Value": st.column_config.TextColumn(
                "Value",
                width="large",
                help="Field value"
            )
        },
        hide_index=True,
        use_container_width=True
    )

def process_company_data(company_name):
    """Process company data and update session state."""
    st.session_state.processing = True
    st.session_state.error_message = None
    
    try:
        # Create company-specific directory
        company_dir = FINANCIAL_DATA_DIR / company_name.replace(" ", "_").lower()
        company_dir.mkdir(exist_ok=True)
        
        # Process financial data
        financial_data, financial_error = asyncio.run(process_financials(company_name))
        if financial_error:
            st.session_state.error_message = financial_error
            logger.error(f"Financial data error: {financial_error}")
            return
            
        # Save financial data
        if financial_data:
            financial_file = company_dir / "financial_data.json"
            with open(financial_file, 'w', encoding='utf-8') as f:
                json.dump(financial_data, f, indent=4, ensure_ascii=False)
            logger.info(f"Saved financial data to {financial_file}")
            
        # Process overview data
        overview_data, overview_error = asyncio.run(process_overview(company_name))
        if overview_error:
            st.session_state.error_message = overview_error
            logger.error(f"Overview data error: {overview_error}")
            return
            
        # Save overview data
        if overview_data:
            overview_file = company_dir / "overview_data.json"
            with open(overview_file, 'w', encoding='utf-8') as f:
                json.dump(overview_data, f, indent=4, ensure_ascii=False)
            logger.info(f"Saved overview data to {overview_file}")
            
        # Update session state
        st.session_state.financial_data = financial_data
        st.session_state.overview_data = overview_data
        
    except Exception as e:
        error_msg = f"Error processing company data: {str(e)}"
        st.session_state.error_message = error_msg
        logger.error(error_msg)
    finally:
        st.session_state.processing = False

def main():
    # Initialize session state for company name if it doesn't exist
    if 'company_name' not in st.session_state:
        st.session_state.company_name = None

    # Sidebar
    with st.sidebar:
        st.markdown('<h1 class="sidebar-title">🏢 Company Research</h1>', unsafe_allow_html=True)
        
        # Instructions in sidebar
        st.markdown('<div class="sidebar-instructions">', unsafe_allow_html=True)
        st.markdown("""
        ### Instructions
        1. Enter the Official company name
        2. Click on "🔍 Research Company" to begin
        3. The results will be displayed in the main screen
        4. You can download the results in CSV format
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Search section
        st.markdown('<div class="sidebar-input">', unsafe_allow_html=True)
        company_name = st.text_input(
            "Company Name",
            placeholder="Enter company name...",
            help="Enter the full company name as registered"
        )
        
        if st.button("🔍 Research Company", disabled=st.session_state.processing):
            if not company_name:
                st.error("Please enter a company name")
            else:
                st.session_state.company_name = company_name
        st.markdown('</div>', unsafe_allow_html=True)

    # Main content
    if st.session_state.company_name:
        st.title(st.session_state.company_name)
        st.markdown("---")
        
        with st.spinner("Researching company information..."):
            process_company_data(st.session_state.company_name)

        # Show error message if any
        if st.session_state.error_message:
            st.error(st.session_state.error_message)

        # Show processing state
        if st.session_state.processing:
            st.info("Processing company data... Please wait.")

        # Create tabs for different data views
        if st.session_state.financial_data or st.session_state.overview_data:
            tab1, tab2, tab3 = st.tabs(["📊 Financial Data", "📋 Company Overview", "📄 Raw Data"])
            
            with tab1:
                if st.session_state.financial_data:
                    display_financial_data(st.session_state.financial_data, "Financial Data")
                else:
                    st.warning("No financial data available")
                    
            with tab2:
                if st.session_state.overview_data:
                    display_nested_data(st.session_state.overview_data, "Company Overview")
                else:
                    st.warning("No company overview data available")
                    
            with tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Financial Data (JSON)")
                    if st.session_state.financial_data:
                        st.json(st.session_state.financial_data)
                    else:
                        st.warning("No financial data available")
                        
                with col2:
                    st.subheader("Company Overview (JSON)")
                    if st.session_state.overview_data:
                        st.json(st.session_state.overview_data)
                    else:
                        st.warning("No company overview data available")

if __name__ == "__main__":
    main()