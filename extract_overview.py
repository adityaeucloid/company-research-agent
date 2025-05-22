import os
import json
import re
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Define the schema for company data and directors using Pydantic
class Director(BaseModel):
    DIN_PAN: str
    Name: str
    Begin_Date: str

class CompanyData(BaseModel):
    CIN: str
    Company_Name: str
    ROC_Code: str
    Registration_Number: str
    Company_Category: str
    Company_SubCategory: str
    Class_of_Company: str
    Authorised_Capital_Rs: str
    Paid_up_Capital_Rs: str
    Number_of_Members: str
    Date_of_Incorporation: str
    Registered_Address: str
    Address_other_than_Registered: str
    Email_Id: str
    Whether_Listed_or_not: str
    Suspended_at_stock_exchange: str
    Date_of_Last_AGM: str
    Date_of_Balance_Sheet: str
    Company_Status: str
    Last_Updated_On: str
    Directors: list[Director]

def parse_model_json(model_output: str) -> dict:
    """
    Extract and parse the JSON object from model_output, removing Markdown code fences.
    """
    no_fences = re.sub(r"```(?:json)?", "", model_output, flags=re.IGNORECASE)
    cleaned = no_fences.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        idx = cleaned.find("{")
        if idx != -1:
            obj, _ = decoder.raw_decode(cleaned[idx:])
            if isinstance(obj, dict):
                return obj
        raise ValueError("No valid JSON object found")

async def extract_company_data(company_name: str, crawled_content: str) -> dict:
    """
    Extract structured company data using Gemini.
    """
    sys_prompt = """
    You are an expert assistant tasked with extracting structured company data from web content.
    You will receive content from multiple sources (zaubacorp.com, indiafilings.com, and falconebiz.com).
    Your task is to:
    1. Extract data from each source separately
    2. Compare the data for consistency
    3. Use the most complete and accurate information
    4. If sources conflict, prefer the more recent or more detailed information
    5. Ensure accuracy, avoid fabricating data, and focus only on the specified company.
    """
    prompt = f"""
    Extract structured data for the company '{company_name}' from the provided content. The content comes from multiple sources (zaubacorp.com, indiafilings.com, and falconebiz.com).
    
    Return a JSON object matching the following schema:
    {{
        "CIN": "string",
        "Company_Name": "string",
        "ROC_Code": "string",
        "Registration_Number": "string",
        "Company_Category": "string",
        "Company_SubCategory": "string",
        "Class_of_Company": "string",
        "Authorised_Capital_Rs": "string",
        "Paid_up_Capital_Rs": "string",
        "Number_of_Members": "string",
        "Date_of_Incorporation": "string",
        "Registered_Address": "string",
        "Address_other_than_Registered": "string",
        "Email_Id": "string",
        "Whether_Listed_or_not": "string",
        "Suspended_at_stock_exchange": "string",
        "Date_of_Last_AGM": "string",
        "Date_of_Balance_Sheet": "string",
        "Company_Status": "string",
        "Last_Updated_On": "string",
        "Directors": [
            {{
                "DIN_PAN": "string",
                "Name": "string",
                "Begin_Date": "string"
            }}
        ]
    }}
    
    Rules:
    - Use "Not Found" for any field where data is unavailable.
    - Ensure dates are in DD-MM-YYYY format.
    - Extract data only for '{company_name}', ignoring similar companies.
    - For each source (zaubacorp, indiafilings, and falconebiz):
        - Extract all available information
        - Note the last updated date if available
        - Assess the completeness of the data
        - Rate your confidence in the data (High/Medium/Low)
    - When sources conflict:
        - Prefer more recent data
        - Prefer more detailed data
        - Prefer data from the source with higher completeness
    - For Directors, include all listed directors with their DIN/PAN, name, and begin date.
    - Include source details to track which information came from where.
    
    Content:
    {crawled_content}
    """

    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Generate content
        response = model.generate_content([sys_prompt, prompt])
        
        if response.text:
            try:
                return parse_model_json(response.text)
            except ValueError as e:
                print(f"Error parsing JSON: {e}")
                return {}
        else:
            print("No content extracted from Gemini")
            return {}
    except Exception as e:
        print(f"Error with Gemini extraction: {e}")
        return {}

async def main():
    """
    Main function to run the company data extraction.
    """
    company_name = "VALUE POINT SYSTEMS PRIVATE LIMITED"
    
    # Read the crawled content from files
    crawled_content = []
    for filename in os.listdir("crawled_content"):
        if filename.startswith(company_name) and filename.endswith(".txt"):
            with open(os.path.join("crawled_content", filename), "r", encoding="utf-8") as f:
                content = f.read()
                crawled_content.append(content)
    
    if not crawled_content:
        print("No crawled content found")
        return
    
    # Extract structured data
    company_data = await extract_company_data(company_name, "\n\n".join(crawled_content))
    if company_data:
        print(f"Extracted data for {company_name}:")
        print(json.dumps(company_data, indent=2))
        
        # Save the extracted data
        with open(f"crawled_content/{company_name}_extracted.json", "w", encoding="utf-8") as f:
            json.dump(company_data, f, indent=2)
    else:
        print("No data extracted")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 