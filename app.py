import streamlit as st
import asyncio
import os
import json
import re
import requests
from urllib.parse import urlparse
from tavily import TavilyClient
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import (
    DomainFilter,
    ContentTypeFilter,
    ContentRelevanceFilter,
    FilterChain,
    URLPatternFilter
)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel
import time
import random

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Define Pydantic models for company data
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

def get_headers():
    """Get random user agent and headers for requests"""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    ]
    return {
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

def extract_cin_from_url(url: str) -> str:
    if "zaubacorp.com" in url:
        parts = url.split("-")
        if len(parts) > 1:
            cin_code = parts[-1]
            if re.match(r'^[A-Z0-9]{21}$', cin_code):
                return cin_code
    return ""

def construct_companycheck_url(company_name: str, cin_code: str) -> str:
    company_slug = company_name.lower().replace(" ", "-")
    return f"https://www.thecompanycheck.com/company/{company_slug}/{cin_code}"

def construct_falconebiz_url(company_name: str, cin_code: str) -> str:
    company_slug = company_name.lower().replace(" ", "-")
    return f"https://www.falconebiz.com/company/{company_slug}-{cin_code}"

def parse_model_json(model_output: str) -> dict:
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

# Crawling functions
async def tavily_search_company(company_name: str, max_results: int = 1) -> list[str]:
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        st.error("TAVILY_API_KEY not found in environment variables")
        return []
    
    client = TavilyClient(api_key=tavily_api_key)
    queries = [
        f"site:zaubacorp.com {company_name}",
        f"site:indiafilings.com {company_name}",
        f"site:falconebiz.com/company {company_name} CIN"
    ]
    
    urls = []
    for query in queries:
        try:
            results = client.search(
                query=query,
                max_results=1,
                search_depth="advanced",
                include_domains=["zaubacorp.com", "indiafilings.com", "falconebiz.com"],
                exclude_domains=["linkedin.com", "facebook.com", "twitter.com"]
            )
            if results["results"]:
                urls.append(results["results"][0]["url"])
        except Exception as e:
            st.warning(f"Tavily search failed for query '{query}': {e}")
    return urls

def crawl_zaubacorp_for_cin(url: str) -> str:
    try:
        response = requests.get(url, headers=get_headers(), timeout=10)
        response.raise_for_status()
        return extract_cin_from_url(url)
    except Exception as e:
        st.warning(f"Error crawling zaubacorp URL {url}: {str(e)}")
        return ""

def crawl_companycheck_data(url: str, company_name: str) -> str:
    try:
        response = requests.get(url, headers=get_headers(), timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        extracted_data = []
        
        company_name_elem = soup.find('h1')
        if company_name_elem:
            extracted_data.append(f"Company Name: {company_name_elem.text.strip()}")
        
        tables = soup.find_all('table')
        for table in tables:
            table_title = table.find_previous(['h2', 'h3', 'h4', 'div'], class_=re.compile(r'title|heading', re.I))
            if table_title:
                extracted_data.append(f"\n{table_title.text.strip()}:")
            for row in table.find_all('tr'):
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    label = cells[0].text.strip()
                    value = cells[1].text.strip()
                    if value and value != "GET PRO" and not value.startswith("₹"):
                        extracted_data.append(f"{label}: {value}")
        
        sections = soup.find_all(['div', 'section'], class_=re.compile(r'financial|balance|profit|loss|revenue|assets|liabilities|charges|registered|details|overview', re.I))
        for section in sections:
            section_title = section.find_previous(['h2', 'h3', 'h4', 'div'], class_=re.compile(r'title|heading', re.I))
            if section_title:
                extracted_data.append(f"\n{section_title.text.strip()}:")
            for p in section.find_all(['p', 'div']):
                text = p.text.strip()
                if text and ":" in text and not text.startswith("₹"):
                    extracted_data.append(text)
        
        company_details = soup.find('div', class_='company-details')
        if company_details:
            extracted_data.append("\nCompany Details:")
            for detail in company_details.find_all(['p', 'div']):
                text = detail.text.strip()
                if text and ":" in text and not text.startswith("₹"):
                    extracted_data.append(text)
        
        if extracted_data:
            os.makedirs("financial_data", exist_ok=True)
            structured_filename = f"financial_data/{company_name}_thecompanycheck_structured.txt"
            with open(structured_filename, "w", encoding="utf-8") as f:
                f.write("\n".join(extracted_data))
            return "\n".join(extracted_data)
        return ""
    except Exception as e:
        st.warning(f"Error crawling {url}: {str(e)}")
        return ""

def crawl_company_data(urls: list[str], company_name: str) -> str:
    markdown_content = []
    os.makedirs("crawled_content", exist_ok=True)
    
    for url in urls:
        try:
            # Add random delay between requests
            time.sleep(random.uniform(1, 3))
            
            response = requests.get(url, headers=get_headers(), timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['header', 'footer', 'form', 'nav', 'script', 'style']):
                element.decompose()
            
            # Extract text content
            content = soup.get_text(separator='\n', strip=True)
            
            if content:
                website_name = urlparse(url).netloc.replace("www.", "").split(".")[0]
                filename = f"crawled_content/{company_name}_{website_name}.txt"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(f"URL: {url}\n\n")
                    f.write(content)
                markdown_content.append(f"Source: {url}\n{content}")
                st.success(f"Successfully crawled {url}")
            else:
                st.warning(f"No content found for {url}")
        except Exception as e:
            st.warning(f"Error crawling {url}: {str(e)}")
    
    return "\n\n---\n\n".join(markdown_content)

# Extraction functions
def extract_financial_metrics(company_name: str, text: str) -> dict:
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""
    You are a financial data extraction assistant. From the provided text, extract the following information for '{company_name}':

    1. Company Name (as a string, extracted from the text or file title)
    2. Financial Year (as a string, e.g., 'FY 2023')
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

    Return the result as a JSON object with four main keys: 'company_name', 'financial_year', 'financial_metrics_YoY_growth', and 'charge_details_in_cr'. Ensure the output is clean and only includes the requested metrics and details. If a metric or detail is not found in the text, include it with a null value. Add '%' symbol to the YOY growth percentages. Add '₹ Cr' symbol to the charge amounts.

    Text:
    {text}

    Example output:
    {{
        "company_name": "Example Company Limited",
        "financial_year": "FY 2023",
        "financial_metrics_YoY_growth": {{
            "Total Revenue": "43.11 %",
            "Revenue from Operations": "43.32 %",
            "Total Assets": "94.39 %",
            "Profit or Loss": "537.46 %",
            "Net Worth": "92.14 %",
            "EBITDA": "227.94 %"
        }},
        "charge_details_in_cr": {{
            "Total Open Charges": "200.00 ₹ Cr",
            "Total Satisfied Charges": "0.00 ₹ Cr",
            "Charges Breakdown by Lending Institution": {{
                "Others": "150.00 ₹ Cr",
                "Hdfc Bank Limited": "50.00 ₹ Cr"
            }},
            "Total Number of Lenders": 2,
            "Top Lender": "Others",
            "Last Charge Activity": "A charge with Others amounted to Rs. 75.00 Cr with Charge ID 100891498 was registered on 04 Mar 2024.",
            "Last Charge Date": "04 Mar 2024",
            "Last Charge Amount": "75.00 ₹ Cr"
        }}
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = re.sub(r'```json\n|\n```', '', response.text).strip()
        return json.loads(response_text)
    except Exception as e:
        st.warning(f"Error extracting financial metrics: {e}")
        return {}

def extract_company_data(company_name: str, crawled_content: str) -> dict:
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""
    Extract structured data for '{company_name}' from the provided content. Return a JSON object matching:
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
            {{"DIN_PAN": "string", "Name": "string", "Begin_Date": "string"}}
        ]
    }}
    Use "Not Found" for unavailable fields. Ensure dates are in DD-MM-YYYY format. Content: {crawled_content}
    """
    
    try:
        response = model.generate_content(prompt)
        return parse_model_json(response.text)
    except Exception as e:
        st.warning(f"Error extracting company data: {e}")
        return {}

# Main pipeline
def run_pipeline(company_name: str) -> tuple[dict, dict]:
    financial_data = {}
    overview_data = {}
    
    # Step 1: Tavily search for URLs
    with st.spinner("Searching for company URLs..."):
        urls = asyncio.run(tavily_search_company(company_name))
        if not urls:
            st.error("No relevant URLs found.")
            return {}, {}
    
    # Step 2: Crawl zaubacorp for CIN
    zaubacorp_url = next((url for url in urls if "zaubacorp.com" in url), None)
    if zaubacorp_url:
        with st.spinner("Extracting CIN from zaubacorp..."):
            cin_code = asyncio.run(crawl_zaubacorp_for_cin(zaubacorp_url))
            if cin_code:
                urls.append(construct_companycheck_url(company_name, cin_code))
                urls.append(construct_falconebiz_url(company_name, cin_code))
    
    # Step 3: Crawl company data (overview)
    with st.spinner("Crawling company data..."):
        crawled_content = asyncio.run(crawl_company_data(urls, company_name))
        if crawled_content:
            overview_data = extract_company_data(company_name, crawled_content)
    
    # Step 4: Crawl financial data from thecompanycheck
    companycheck_url = next((url for url in urls if "thecompanycheck.com" in url), None)
    if companycheck_url:
        with st.spinner("Crawling financial data..."):
            financial_content = asyncio.run(crawl_companycheck_data(companycheck_url, company_name))
            if financial_content:
                financial_data = extract_financial_metrics(company_name, financial_content)
    
    return overview_data, financial_data

# Streamlit UI
st.title("Company Data Extraction")
st.write("Enter a company name to extract overview and financial data.")

company_name = st.text_input("Company Name", value="VALUE POINT SYSTEMS PRIVATE LIMITED")
if st.button("Run Pipeline"):
    if not company_name:
        st.error("Please enter a company name.")
    else:
        if not os.getenv("TAVILY_API_KEY") or not os.getenv("GEMINI_API_KEY"):
            st.error("Please ensure TAVILY_API_KEY and GEMINI_API_KEY are set in your .env file.")
        else:
            overview_data, financial_data = run_pipeline(company_name)
            
            st.subheader("Company Overview")
            if overview_data:
                st.json(overview_data)
            else:
                st.warning("No overview data extracted.")
            
            st.subheader("Financial Data")
            if financial_data:
                st.json(financial_data)
            else:
                st.warning("No financial data extracted.")

if __name__ == "__main__":
    st.write("Note: Ensure TAVILY_API_KEY and GEMINI_API_KEY are set in your .env file.")