import streamlit as st
import asyncio
import json
import os
import pandas as pd
from dotenv import load_dotenv
from combined_financials import main as financials_main, COMPANY_NAME
from detailed_overview import CompanyOverviewGenerator
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('streamlit_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Streamlit app configuration
st.set_page_config(
    page_title="LumenAI Researcher",
    page_icon="🔍",
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
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: white;
        color: black;
        border: 1px solid #ddd;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #f0f0f0;
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
        'Group Companies': [],
        'Joint Ventures': []
    }
    
    main_data = data.copy()
    
    # Extract expandable sections
    for section in expandable_sections.keys():
        if section in data:
            expandable_sections[section] = data[section]
            del main_data[section]
    
    return main_data, expandable_sections

def display_nested_data(data, section_name):
    """Display data in a single, well-organized table with expandable sections."""
    if not data:
        return
    
    # Extract expandable sections
    main_data, expandable_sections = extract_expandable_sections(data)
    
    # Flatten the main data
    flat_data = flatten_dict(main_data)
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'Field': flat_data.keys(),
        'Value': flat_data.values()
    })
    
    # Sort fields for better organization
    df = df.sort_values('Field')
    
    # Display the main table
    st.markdown(f"### {section_name}")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
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
        }
    )
    
    # Display expandable sections
    for section_name, section_data in expandable_sections.items():
        if section_data:
            if isinstance(section_data, list) and all(isinstance(item, dict) for item in section_data):
                # Create expander for the section
                with st.expander(f"📋 {section_name}", expanded=True):
                    # Convert list of dictionaries to DataFrame
                    df = pd.DataFrame(section_data)
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            col: st.column_config.TextColumn(
                                col,
                                width="medium",
                                help=f"{col} information"
                            ) for col in df.columns
                        }
                    )

def process_company_data(company_name):
    """Process company data and return the results."""
    try:
        # Update COMPANY_NAME in combined_financials
        globals()['COMPANY_NAME'] = company_name.upper()

        # Run financials extraction
        try:
            asyncio.run(financials_main())
            financial_file = f"financial_data/{company_name}_extracted_financial_data.json"
            if os.path.exists(financial_file):
                with open(financial_file, 'r', encoding='utf-8') as f:
                    financial_data = json.load(f)
            else:
                financial_data = {}
                st.warning("No financial data found for the company.")
        except Exception as e:
            logger.error(f"Error fetching financial data: {str(e)}")
            st.error(f"Error fetching financial data: {str(e)}")
            financial_data = {}

        # Run company overview generation
        try:
            generator = CompanyOverviewGenerator()
            overview_data = generator.get_overview_report(company_name)
            clean_name = company_name.replace(" ", "_").lower()
            overview_file = f"{clean_name}_overview.json"
            generator.save_report(overview_data, company_name, overview_file)
        except Exception as e:
            logger.error(f"Error fetching company overview: {str(e)}")
            st.error(f"Error fetching company overview: {str(e)}")
            overview_data = {}

        return financial_data, overview_data

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")
        return {}, {}

def main():
    # Initialize session state for company name if it doesn't exist
    if 'company_name' not in st.session_state:
        st.session_state.company_name = None

    # Sidebar
    with st.sidebar:
        st.markdown('<h1 class="sidebar-title">🔍 LumenAI Researcher</h1>', unsafe_allow_html=True)
        
        # Instructions in sidebar
        st.markdown('<div class="sidebar-instructions">', unsafe_allow_html=True)
        st.markdown("""
        ### Instructions
        1. Enter the Official company name
        2. Click on "Start Search" to begin
        3. The results will be displayed in the main screen
        4. You can download the results in CSV format
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Search section
        st.markdown('<div class="sidebar-input">', unsafe_allow_html=True)
        company_name = st.text_input(
            "Company Name",
            help="Enter the complete official company name"
        )
        
        if st.button("Start Search", use_container_width=True):
            if not company_name:
                st.error("Please enter a company name.")
            else:
                st.session_state.company_name = company_name
        st.markdown('</div>', unsafe_allow_html=True)

    # Main content
    if st.session_state.company_name:
        st.title(st.session_state.company_name)
        st.markdown("---")
        
        with st.spinner("Researching company information..."):
            financial_data, overview_data = process_company_data(st.session_state.company_name)

            # Create tabs
            tab1, tab2 = st.tabs(["Company Overview", "Financial Data"])

            with tab1:
                if overview_data:
                    # Display overview data in a single table with expandable sections
                    display_nested_data(overview_data, "Company Overview")
                    # Download button for overview data
                    overview_df = convert_to_dataframe(overview_data)
                    csv = overview_df.to_csv(index=False)
                    st.download_button(
                        label="Download Overview Data as CSV",
                        data=csv,
                        file_name=f"{st.session_state.company_name.replace(' ', '_').lower()}_overview.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.warning("No company overview available.")

            with tab2:
                if financial_data:
                    # Display financial data in a single table with expandable sections
                    display_nested_data(financial_data, "Financial Data")
                    # Download button for financial data
                    financial_df = convert_to_dataframe(financial_data)
                    csv = financial_df.to_csv(index=False)
                    st.download_button(
                        label="Download Financial Data as CSV",
                        data=csv,
                        file_name=f"{st.session_state.company_name.replace(' ', '_').lower()}_financials.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.warning("No financial data available.")

if __name__ == "__main__":
    main()