import os
from dotenv import load_dotenv
import google.generativeai as genai
import re
import json

# Hardcoded file path
FILE_PATH = r"C:\Codes\company-research-agent\financial_data\VALUE POINT SYSTEMS PRIVATE LIMITED_thecompanycheck_structured.txt"

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.0-flash')

def read_file(file_path):
    """
    Read the content of the file at the given path.
    Returns the text content or raises an error if the file cannot be read.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        raise
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        raise

def extract_financial_metrics(text):
    """
    Extract financial metrics and charge-related details from the text using Gemini API.
    Returns a dictionary with company name, financial year, metrics, and charge details.
    """
    # Prepare the prompt for Gemini
    prompt = """
    You are a financial data extraction assistant. From the provided text, extract the following information for the company described in the text:

    1. Company Name (as a string, extracted from the text or file title)
    2. Financial Year (as a string, e.g., 'FY 2023', for the year associated with the financial metrics)
    3. Financial Metrics with their Year-over-Year (YOY) growth percentages (as floats):
       - Total Revenue
       - Revenue from Operations
       - Total Assets
       - Profit or Loss
       - Net Worth
       - EBITDA
    4. Charge-related details (mention the unit of the charge amounts):
       - Total Open Charges (in ₹ Cr, as a float)
       - Total Satisfied Charges (in ₹ Cr, as a float)
       - Charges Breakdown by Lending Institution (as a dictionary with lender names as keys and amounts in ₹ Cr as floats)
       - Total Number of Lenders (count of unique lenders in the charges breakdown)
       - Top Lender (the lender with the highest charge amount, or the first lender alphabetically if tied)
       - Last Charge Activity (description of the most recent charge activity, as a string)
       - Last Charge Date (date of the most recent charge, in DD MMM YYYY format)
       - Last Charge Amount (amount of the most recent charge in ₹ Cr, as a float)

    Return the result as a JSON object with four main keys: 'company_name' for the company name, 'financial_year' for the financial year, 'financial_metrics_YoY_growth' for the YOY growth metrics, and 'charge_details_in_cr' for the charge-related information. Ensure the output is clean and only includes the requested metrics and details. If a metric or detail is not found in the text, include it with a null value. Add '%' symbol to the YOY growth percentages. Add '₹ Cr' symbol to the charge amounts.

    Text:
    {text}

    Example output:
    ```json
    {{
        "company_name": "Example Company Limited",
        "financial_year": "FY 2023",
        "financial_metrics_YoY_growth": {{
            "Total Revenue": 43.11 %,
            "Revenue from Operations": 43.32 %,
            "Total Assets": 94.39 %,
            "Profit or Loss": 537.46 %,
            "Net Worth": 92.14 %,
            "EBITDA": 227.94 %
        }},
        "charge_details_in_cr": {{
            "Total Open Charges": 200.00 ₹ Cr,
            "Total Satisfied Charges": 0.00 ₹ Cr,
            "Charges Breakdown by Lending Institution": {{
                "Others": 150.00 ₹ Cr,
                "Hdfc Bank Limited": 50.00
            }},
            "Total Number of Lenders": 2,
            "Top Lender": "Others",
            "Last Charge Activity": "A charge with Others amounted to Rs. 75.00 Cr with Charge ID 100891498 was registered on 04 Mar 2024.",
            "Last Charge Date": "04 Mar 2024",
            "Last Charge Amount": 75.00 ₹ Cr
        }}
    }}
    ```
    """.format(text=text)

    try:
        # Call the Gemini API
        response = model.generate_content(prompt)
        response_text = response.text

        # Clean the response (remove markdown code fences if present)
        response_text = re.sub(r'```json\n|\n```', '', response_text).strip()

        # Parse the JSON response
        extracted_data = json.loads(response_text)
        return extracted_data

    except Exception as e:
        print(f"Error processing with Gemini API: {e}")
        return {}

def main():
    # Read the file
    try:
        scraped_text = read_file(FILE_PATH)
    except Exception:
        return

    # Extract metrics and charge details using Gemini
    extracted_data = extract_financial_metrics(scraped_text)

    # Print the extracted data
    if extracted_data:
        print(json.dumps(extracted_data, indent=4))
        # Save the extracted data to a JSON file
        output_file = "extracted_financial_data.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=4, ensure_ascii=False)
        print(f"Extracted financial data saved to {output_file}")
    else:
        print("Failed to extract financial metrics and charge details.")

if __name__ == "__main__":
    main()