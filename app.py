import streamlit as st
import asyncio
import json
import os
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
st.set_page_config(page_title="Company Financials and Overview", layout="wide")

# Title and description
st.title("Company Financials and Overview App")
st.markdown("""
Enter a company name to retrieve its financial metrics and detailed overview.
The app fetches financial data from thecompanycheck.com and generates a comprehensive overview using web search.
""")

# Input field for company name
company_name = st.text_input("Enter Company Name", value="VALUE POINT SYSTEMS PRIVATE LIMITED")

# Button to trigger data fetching
if st.button("Fetch Data"):
    if not company_name:
        st.error("Please enter a company name.")
    else:
        with st.spinner("Fetching financial data and company overview..."):
            try:
                # Update COMPANY_NAME in combined_financials
                globals()['COMPANY_NAME'] = company_name.upper()

                # Run financials extraction
                financial_data = None
                try:
                    # Since financials_main is async, run it in Streamlit's async context
                    financial_data = asyncio.run(financials_main())
                    # Read the saved financial data
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

                # Run company overview generation
                overview_data = None
                try:
                    generator = CompanyOverviewGenerator()
                    overview_data = generator.get_overview_report(company_name)
                    # Save overview report
                    clean_name = company_name.replace(" ", "_").lower()
                    overview_file = f"{clean_name}_overview.json"
                    generator.save_report(overview_data, company_name, overview_file)
                except Exception as e:
                    logger.error(f"Error fetching company overview: {str(e)}")
                    st.error(f"Error fetching company overview: {str(e)}")

                # Display results in two columns
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Financial Data")
                    if financial_data:
                        st.json(financial_data)
                        st.success(f"Financial data saved to {financial_file}")
                    else:
                        st.warning("No financial data available.")

                with col2:
                    st.subheader("Company Overview")
                    if overview_data:
                        st.json(overview_data)
                        st.success(f"Company overview saved to {overview_file}")
                    else:
                        st.warning("No company overview available.")

            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                st.error(f"An unexpected error occurred: {str(e)}")

# Instructions for running the app
st.markdown("""
### Instructions
1. Ensure you have set the required API keys (`TAVILY_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`) in a `.env` file.
2. The `combined_financials.py` and `detailed_overview.py` files must be in the same directory as this app.
3. Install required dependencies: `streamlit`, `python-dotenv`, `tavily`, `crawl4ai`, `google-generativeai`, `openai`, `beautifulsoup4`.
4. Run the app using `streamlit run app.py`.
""")