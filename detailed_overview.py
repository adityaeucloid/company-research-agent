from openai import OpenAI
from dotenv import load_dotenv
import os
import re
import json
import logging
from json import JSONDecoder, JSONDecodeError
from typing import Dict, Optional, Any
from datetime import datetime
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('company_overview.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

class CompanyOverviewGenerator:
    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3, retry_delay: int = 2):
        """
        Initialize the Company Overview Generator.
        
        Args:
            api_key: OpenAI API key. If None, will use environment variable.
            max_retries: Maximum number of retries for API calls.
            retry_delay: Delay between retries in seconds.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Default structure for company overview
        self.default_structure = {
            # Basic Company Information
            "CIN Number": "Not Found",
            "Name": "Not Found", 
            "Company Type": "Not Found",  # Private/Public/LLP etc.
            "Listed on Stock Exchange": "Not Found",
            "Stock Exchange": "Not Found",  # BSE/NSE/Both
            "Stock Symbol/Ticker": "Not Found",
            "Company Status (for eFiling)": "Not Found",
            "PAN": "Not Found",
            "TAN": "Not Found",  # Tax Deduction Account Number
            "GSTIN": "Not Found",  # GST Identification Number
            "Date of Incorporation": "Not Found",
            "Age of Company (Years)": "Not Found",
            
            # Legal & Regulatory Information
            "LEI Number": "Not Found",
            "LEI Expiry Date": "Not Found", 
            "RoC Code": "Not Found",
            "Registrar of Companies": "Not Found",
            "MCA Compliance Rating": "Not Found",
            "Annual Filing Status": "Not Found",
            
            # Contact Information
            "email id": "Not Found",
            "Phone Number": "Not Found",
            "Fax Number": "Not Found",
            "website": "Not Found",
            
            # Address Information
            "Registered Address": "Not Found",
            "Corporate Address": "Not Found", 
            
            # Directors & Key Personnel
            "Current Directors & Key Managerial Personnel": [],
            "Total Number of Directors": "Not Found",
            
            # Business Information
            "Sector": "Not Found",
            "Industry": "Not Found", 
            "Sub-Industry": "Not Found",
            "Business Activity": "Not Found",
            "Main Objects of Company": "Not Found",
            "Number of Employees": "Not Found",
            "Employee Range": "Not Found",  # 1-10, 11-50, etc.
            
            # Financial Information
            "Authorised Share Capital": "Not Found",
            "Paid-up Share Capital": "Not Found",
            "Current Share Capital": "Not Found",
            "Date of Last Filed Balance Sheet": "Not Found",
            "Financial Year End": "Not Found",
            
            # Shareholding Information
            "Number of Shareholders": "Not Found",
            "Promoter Shareholding %": "Not Found",
            "Public Shareholding %": "Not Found",
            
            # Subsidiary & Group Information
            "Parent Company": "Not Found",
            "Subsidiary Companies": []
            
            
        }

    def parse_model_json(self, model_output: str) -> Dict[str, Any]:
        """
        Extract and parse the first valid JSON object found in model_output.
        
        Args:
            model_output: Raw output from the model
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            ValueError: If no valid JSON object is found
        """
        try:
            # Remove Markdown code fences
            no_fences = re.sub(r"```(?:json)?", "", model_output, flags=re.IGNORECASE)
            cleaned = no_fences.strip()
            
            # Try direct JSON parsing first
            try:
                return json.loads(cleaned)
            except JSONDecodeError:
                pass
            
            # Use JSONDecoder.raw_decode to find the first valid JSON object
            decoder = JSONDecoder()
            idx = 0
            length = len(cleaned)
            
            while idx < length:
                brace_pos = cleaned.find("{", idx)
                if brace_pos == -1:
                    break
                    
                try:
                    obj, end = decoder.raw_decode(cleaned[brace_pos:])
                    logger.info("Successfully parsed JSON from model output")
                    return obj
                except JSONDecodeError:
                    idx = brace_pos + 1
                    continue
            
            raise ValueError("No valid JSON object found in model output")
            
        except Exception as e:
            logger.error(f"Error parsing JSON: {str(e)}")
            raise ValueError(f"JSON parsing failed: {str(e)}")

    def create_comprehensive_prompt(self, company_name: str) -> str:
        """
        Create a comprehensive prompt for gathering company information.
        
        Args:
            company_name: Name of the company to research
            
        Returns:
            Formatted prompt string
        """
        json_structure = json.dumps(self.default_structure, indent=2)
        
        prompt = f"""
You are a comprehensive business intelligence assistant with live web search capabilities. You will research detailed information about a company and provide a structured JSON response.

Company to research: {company_name}

Please search the web extensively and gather information for ALL the following fields. Return the response in the exact JSON structure provided below:

{json_structure}

IMPORTANT INSTRUCTIONS:
1. Search multiple authoritative sources including:
   - Ministry of Corporate Affairs (MCA) database
   - Stock exchange websites (BSE/NSE)
   - Company's official website
   - Financial databases
   - Business news sources
   - LinkedIn and other professional networks

2. For fields you cannot find information, use "Not Found" as the value
3. For array fields (like directors, subsidiaries), provide empty arrays [] if no information is found
4. Ensure all dates are in DD/MM/YYYY or DD-MM-YYYY format
5. For financial figures, include currency (₹ for Indian companies)
6. Verify information from multiple sources when possible
7. Focus on the most recent and accurate information available

8. For the "Current Directors & Key Managerial Personnel" field, include:
   - DIN (Director Identification Number)
   - Director Name
   - Designation (Chairman, MD, CEO, CFO, etc.)
   - Appointment Date
   - Shareholding % (if applicable)

9. Return ONLY the JSON response, no additional commentary or explanations.

The JSON must be valid and well-formatted. Double-check all syntax before responding.
"""
        return prompt

    def get_overview_report(self, company_name: str) -> Dict[str, Any]:
        """
        Get comprehensive overview report of a company with retry logic.
        
        Args:
            company_name: Name of the company to research
            
        Returns:
            Dictionary containing company information
        """
        logger.info(f"Starting overview report generation for: {company_name}")
        
        for attempt in range(self.max_retries):
            try:
                prompt = self.create_comprehensive_prompt(company_name)
                
                logger.info(f"Making API call (attempt {attempt + 1}/{self.max_retries})")
                
                completion = self.client.chat.completions.create(
                    model="gpt-4o-search-preview",
                    web_search_options={
                        "search_context_size": "high",
                        "user_location": {
                            "type": "approximate", 
                            "approximate": {
                                "country": "IN"
                            }
                        }
                    },
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    max_tokens=4000
                )
                
                response = completion.choices[0].message.content
                logger.info("API call successful")
                
                # Parse and validate the JSON response
                report = self.parse_model_json(response)
                
                # Merge with default structure to ensure all fields are present
                final_report = {**self.default_structure, **report}
                
                
                
                logger.info(f"Successfully generated overview report for: {company_name}")
                return final_report
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed for {company_name}")
                    
                    # Return default structure with error information
                    error_report = self.default_structure.copy()
                    error_report["_metadata"] = {
                        "generated_at": datetime.now().isoformat(),
                        "company_searched": company_name,
                        "error": str(e),
                        "status": "failed"
                    }
                    return error_report

    def save_report(self, report: Dict[str, Any], company_name: str, filename: Optional[str] = None) -> str:
        """
        Save the report to a JSON file.
        
        Args:
            report: Company report dictionary
            company_name: Original company name searched
            filename: Output filename (optional)
            
        Returns:
            Filename of the saved report
        """
        if not filename:
            # Clean company name for filename
            clean_name = re.sub(r'[^\w\s-]', '', company_name).strip()
            clean_name = re.sub(r'[-\s]+', '_', clean_name).lower()
            filename = f"{clean_name}_overview.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Report saved to: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            raise

def main():
    """
    Main function to demonstrate usage.
    """
    try:
        # Initialize the generator
        generator = CompanyOverviewGenerator()
        
        # List of companies to analyze
        companies = [
            "Value Point Systems Private Limited",
            "Riota Private Limited",
            # Add more companies as needed
        ]
        
        for company_name in companies:
            logger.info(f"Processing: {company_name}")
            
            # Generate report
            report = generator.get_overview_report(company_name)
            
            # Save report
            filename = generator.save_report(report, company_name)
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"COMPANY OVERVIEW REPORT: {company_name}")
            print(f"{'='*60}")
            print(f"CIN: {report.get('CIN Number', 'Not Found')}")
            print(f"Status: {report.get('Company Status (for eFiling)', 'Not Found')}")
            print(f"Sector: {report.get('Sector', 'Not Found')}")
            print(f"Website: {report.get('website', 'Not Found')}")
            print(f"Report saved to: {filename}")
            print(f"{'='*60}\n")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()