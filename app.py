import streamlit as st
import asyncio
import json
import os
import pandas as pd
from dotenv import load_dotenv
from combined_financials import main as financials_main
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

def display_financial_data(data, section_name):
    """Display financial data in organized sections."""
    if not data:
        return
    
    st.markdown(f"### {section_name}")
    
    # Organize financial data
    organized_data = organize_financial_data(data)
    
    # Display Charge Details section
    if organized_data['Charge Details']:
        st.markdown("#### Charge Details")
        charge_data = organized_data['Charge Details']
        
        # Extract charges breakdown
        charges_breakdown = charge_data.pop('Charges Breakdown by Lending Institution', {})
        
        # Display main charge metrics
        charge_df = pd.DataFrame({
            'Metric': charge_data.keys(),
            'Value': [convert_to_string(v) for v in charge_data.values()]
        })
        st.dataframe(
            charge_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn(
                    "Metric",
                    width="medium",
                    help="Charge metric"
                ),
                "Value": st.column_config.TextColumn(
                    "Value",
                    width="large",
                    help="Metric value"
                )
            }
        )
        
        # Display charges breakdown in expandable section
        if charges_breakdown:
            with st.expander("📊 Charges Breakdown by Lending Institution", expanded=True):
                breakdown_df = pd.DataFrame({
                    'Lender': charges_breakdown.keys(),
                    'Amount': [convert_to_string(v) for v in charges_breakdown.values()]
                })
                st.dataframe(
                    breakdown_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Lender": st.column_config.TextColumn(
                            "Lender",
                            width="medium",
                            help="Lending institution"
                        ),
                        "Amount": st.column_config.TextColumn(
                            "Amount",
                            width="medium",
                            help="Charge amount"
                        )
                    }
                )
    
    # Display Financial Metrics YoY section
    if organized_data['Financial Metrics YoY']:
        st.markdown("#### Financial Metrics (YoY Growth)")
        metrics_df = pd.DataFrame({
            'Metric': organized_data['Financial Metrics YoY'].keys(),
            'Growth': [convert_to_string(v) for v in organized_data['Financial Metrics YoY'].values()]
        })
        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn(
                    "Metric",
                    width="medium",
                    help="Financial metric"
                ),
                "Growth": st.column_config.TextColumn(
                    "Growth",
                    width="medium",
                    help="Year-over-Year growth"
                )
            }
        )
    
    # Display other information
    if organized_data['Other Information']:
        st.markdown("#### Other Information")
        other_df = pd.DataFrame({
            'Field': organized_data['Other Information'].keys(),
            'Value': [convert_to_string(v) for v in organized_data['Other Information'].values()]
        })
        st.dataframe(
            other_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Field": st.column_config.TextColumn(
                    "Field",
                    width="medium",
                    help="Information field"
                ),
                "Value": st.column_config.TextColumn(
                    "Value",
                    width="large",
                    help="Field value"
                )
            }
        )

def display_nested_data(data, section_name):
    """Display data in a single, well-organized table with expandable sections."""
    if not data:
        return
    
    # Extract expandable sections
    main_data, expandable_sections = extract_expandable_sections(data)
    
    # Flatten the main data
    flat_data = flatten_dict(main_data)
    
    # Convert to DataFrame with string values
    df = pd.DataFrame({
        'Field': flat_data.keys(),
        'Value': [convert_to_string(v) for v in flat_data.values()]
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
            if section_name == 'GST Details':
                # Create expander for GST Details
                with st.expander(f"📋 {section_name}", expanded=True):
                    # Convert list of dictionaries to DataFrame
                    gst_df = pd.DataFrame(section_data)
                    st.dataframe(
                        gst_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "State": st.column_config.TextColumn(
                                "State",
                                width="medium",
                                help="State of GST registration"
                            ),
                            "GSTIN": st.column_config.TextColumn(
                                "GSTIN",
                                width="medium",
                                help="GST Identification Number"
                            )
                        }
                    )
            elif isinstance(section_data, list) and all(isinstance(item, dict) for item in section_data):
                # Create expander for other sections
                with st.expander(f"📋 {section_name}", expanded=True):
                    # Convert list of dictionaries to DataFrame with string values
                    df = pd.DataFrame([
                        {k: convert_to_string(v) for k, v in item.items()}
                        for item in section_data
                    ])
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
        # Run financials extraction
        try:
            asyncio.run(financials_main(company_name))
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
                    # Display financial data in organized sections
                    display_financial_data(financial_data, "Financial Data")
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