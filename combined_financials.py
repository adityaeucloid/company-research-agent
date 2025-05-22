import asyncio
import os
import re
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
from crawl4ai.content_filter_strategy import PruningContentFilter
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import google.generativeai as genai
import json
import time

# Define the company name at the top
COMPANY_NAME = "VALUE POINT SYSTEMS PRIVATE LIMITED"

# Load environment variables
load_dotenv()

def extract_cin_from_url(url: str) -> str:
    """
    Extract CIN code from zaubacorp.com URL.
    """
    if "zaubacorp.com" in url:
        parts = url.split("-")
        if len(parts) > 1:
            cin_code = parts[-1]
            if re.match(r'^[A-Z0-9]{21}$', cin_code):
                return cin_code
    return ""

def construct_companycheck_url(company_name: str, cin_code: str) -> str:
    """
    Construct a valid thecompanycheck.com URL with company name and CIN.
    """
    company_slug = company_name.lower().replace(" ", "-")
    return f"https://www.thecompanycheck.com/company/{company_slug}/{cin_code}"

def construct_falconebiz_url(company_name: str, cin_code: str) -> str:
    """
    Construct a valid falconebiz.com URL with company name and CIN.
    """
    company_slug = company_name.replace(" ", "-")
    return f"https://www.falconebiz.com/company/{company_slug}-{cin_code}"

async def tavily_search_company(company_name: str, max_results: int = 1) -> list[str]:
    """
    Perform a Tavily search for zaubacorp.com URL of the company.
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables")
    
    client = TavilyClient(api_key=tavily_api_key)
    query = f"site:zaubacorp.com {company_name}"
    
    try:
        results = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_domains=["zaubacorp.com"],
            exclude_domains=["linkedin.com", "facebook.com", "twitter.com"]
        )
        urls = [result["url"] for result in results["results"]]
        if urls:
            print(f"Found zaubacorp URL: {urls[0]}")
        else:
            print("No zaubacorp URL found")
        return urls
    except Exception as e:
        print(f"Tavily search failed for query '{query}': {e}")
        return []

async def crawl_zaubacorp_for_cin(url: str) -> str:
    """
    Crawl zaubacorp.com URL to extract CIN only.
    """
    browser_config = BrowserConfig(
        headless=True,
        browser_type="chromium"
    )

    filter_chain = FilterChain([
        DomainFilter(
            allowed_domains=["zaubacorp.com"],
            blocked_domains=[]
        ),
        ContentTypeFilter(allowed_types=["text/html"])
    ])

    run_config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=0,
            include_external=False,
            max_pages=1,
            filter_chain=filter_chain
        ),
        cache_mode='BYPASS',
        verbose=True
    )

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)
            if isinstance(result, list) and result:
                cin_code = extract_cin_from_url(url)
                return cin_code
            print(f"Failed to process zaubacorp URL: {url}")
            return ""
    except Exception as e:
        print(f"Error crawling zaubacorp URL {url}: {str(e)}")
        return ""

async def crawl_companycheck_data(url: str, company_name: str) -> str:
    """
    Crawl thecompanycheck.com URL and return concatenated markdown content.
    """
    browser_config = BrowserConfig(
        headless=True,
        browser_type="chromium",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )

    filter_chain = FilterChain([
        DomainFilter(
            allowed_domains=["thecompanycheck.com"],
            blocked_domains=[]
        ),
        URLPatternFilter(
            patterns=[url]
        ),
        ContentTypeFilter(allowed_types=["text/html"]),
        ContentRelevanceFilter(
            query=f"financial data {company_name}",
            threshold=0.5
        )
    ])

    keyword_scorer = KeywordRelevanceScorer(
        keywords=[
            "company", "financial", "directors", "cin", "incorporation",
            "capital", "balance sheet", "annual report", "registration",
            "roc", "status", "address", "email", company_name.lower()
        ],
        weight=0.8
    )

    run_config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=0,
            include_external=False,
            max_pages=1,
            filter_chain=filter_chain,
            url_scorer=keyword_scorer
        ),
        excluded_tags=['header', 'footer', 'form', 'nav', 'script', 'style'],
        cache_mode='BYPASS',
        verbose=True,
        extraction_strategy=LLMExtractionStrategy(
            extract_metadata=True,
            extract_links=False
        )
    )

    os.makedirs("financial_data", exist_ok=True)
    structured_filename = f"financial_data/{company_name}_thecompanycheck_structured.txt"

    async with AsyncWebCrawler(config=browser_config) as crawler:
        try:
            print(f"\nStarting crawl for URL: {url}")
            result = await crawler.arun(url=url, config=run_config)
            
            print("\nCrawl completed. Analyzing results...")
            if isinstance(result, list) and result:
                item = result[0]
                print(f"\nItem type: {type(item)}")
                
                if hasattr(item, 'success'):
                    print(f"Success status: {item.success}")
                
                content = None
                
                content_attrs = [
                    'html', 'cleaned_html', 'fit_html', 'extracted_content',
                    'markdown', 'markdown_v2', 'fit_markdown'
                ]
                
                for attr in content_attrs:
                    if hasattr(item, '_results') and isinstance(item._results, list) and item._results:
                        if hasattr(item._results[0], attr):
                            content = getattr(item._results[0], attr)
                            print(f"\nFound content in {attr}")
                            print(f"Content length: {len(content) if content else 0}")
                            if content:
                                break
                
                if not content and hasattr(item, 'markdown'):
                    if hasattr(item.markdown, 'fit_markdown'):
                        content = item.markdown.fit_markdown
                        print("\nExtracted content from fit_markdown")
                    elif hasattr(item.markdown, 'content'):
                        content = item.markdown.content
                        print("\nExtracted content from markdown.content")
                
                if content:
                    print(f"\nFinal content length: {len(content)}")
                    
                    soup = BeautifulSoup(content, 'html.parser')
                    extracted_data = []
                    
                    company_name_elem = soup.find('h1')
                    if company_name_elem:
                        extracted_data.append(f"Company Name: {company_name_elem.text.strip()}")
                    
                    tables = soup.find_all('table')
                    print(f"\nFound {len(tables)} tables")
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
                    print(f"\nFound {len(sections)} sections")
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
                        print("\nFound company-details section")
                        extracted_data.append("\nCompany Details:")
                        for detail in company_details.find_all(['p', 'div']):
                            text = detail.text.strip()
                            if text and ":" in text and not text.startswith("₹"):
                                extracted_data.append(text)
                    
                    if extracted_data:
                        with open(structured_filename, "w", encoding="utf-8") as f:
                            f.write("\n".join(extracted_data))
                        print(f"\nSuccessfully saved structured content to {structured_filename}")
                        return "\n".join(extracted_data)
                    else:
                        print("\nNo structured data extracted from the page")
                else:
                    print("\nNo content found in the crawled page")
            else:
                print("\nNo results returned from crawler")
                
        except Exception as e:
            print(f"\nError crawling {url}: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
    
    return ""

async def crawl_falconebiz_data(url: str, company_name: str) -> str:
    """
    Crawl falconebiz.com URL and return concatenated markdown content.
    """
    browser_config = BrowserConfig(
        headless=True,
        browser_type="chromium",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )

    filter_chain = FilterChain([
        DomainFilter(
            allowed_domains=["falconebiz.com"],
            blocked_domains=[]
        ),
        URLPatternFilter(
            patterns=[url]
        ),
        ContentTypeFilter(allowed_types=["text/html"]),
        ContentRelevanceFilter(
            query=f"financial data {company_name}",
            threshold=0.5
        )
    ])

    keyword_scorer = KeywordRelevanceScorer(
        keywords=[
            "company", "financial", "directors", "cin", "incorporation",
            "capital", "balance sheet", "annual report", "registration",
            "roc", "status", "address", "email", company_name.lower()
        ],
        weight=0.8
    )

    run_config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=0,
            include_external=False,
            max_pages=1,
            filter_chain=filter_chain,
            url_scorer=keyword_scorer
        ),
        excluded_tags=['header', 'footer', 'form', 'nav', 'script', 'style'],
        cache_mode='BYPASS',
        verbose=True,
        extraction_strategy=LLMExtractionStrategy(
            extract_metadata=True,
            extract_links=False
        )
    )

    os.makedirs("financial_data", exist_ok=True)
    structured_filename = f"financial_data/{company_name}_falconebiz_structured.txt"

    async with AsyncWebCrawler(config=browser_config) as crawler:
        try:
            print(f"\nStarting crawl for falconebiz URL: {url}")
            result = await crawler.arun(url=url, config=run_config)
            
            print("\nCrawl completed. Analyzing results...")
            if isinstance(result, list) and result:
                item = result[0]
                print(f"\nItem type: {type(item)}")
                
                if hasattr(item, 'success'):
                    print(f"Success status: {item.success}")
                
                content = None
                
                content_attrs = [
                    'html', 'cleaned_html', 'fit_html', 'extracted_content',
                    'markdown', 'markdown_v2', 'fit_markdown'
                ]
                
                for attr in content_attrs:
                    if hasattr(item, '_results') and isinstance(item._results, list) and item._results:
                        if hasattr(item._results[0], attr):
                            content = getattr(item._results[0], attr)
                            print(f"\nFound content in {attr}")
                            print(f"Content length: {len(content) if content else 0}")
                            if content:
                                break
                
                if not content and hasattr(item, 'markdown'):
                    if hasattr(item.markdown, 'fit_markdown'):
                        content = item.markdown.fit_markdown
                        print("\nExtracted content from fit_markdown")
                    elif hasattr(item.markdown, 'content'):
                        content = item.markdown.content
                        print("\nExtracted content from markdown.content")
                
                if content:
                    print(f"\nFinal content length: {len(content)}")
                    
                    soup = BeautifulSoup(content, 'html.parser')
                    extracted_data = []
                    
                    company_name_elem = soup.find('h1')
                    if company_name_elem:
                        extracted_data.append(f"Company Name: {company_name_elem.text.strip()}")
                    
                    tables = soup.find_all('table')
                    print(f"\nFound {len(tables)} tables")
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
                    print(f"\nFound {len(sections)} sections")
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
                        print("\nFound company-details section")
                        extracted_data.append("\nCompany Details:")
                        for detail in company_details.find_all(['p', 'div']):
                            text = detail.text.strip()
                            if text and ":" in text and not text.startswith("₹"):
                                extracted_data.append(text)
                    
                    if extracted_data:
                        with open(structured_filename, "w", encoding="utf-8") as f:
                            f.write("\n".join(extracted_data))
                        print(f"\nSuccessfully saved structured content to {structured_filename}")
                        return "\n".join(extracted_data)
                    else:
                        print("\nNo structured data extracted from the page")
                else:
                    print("\nNo content found in the crawled page")
            else:
                print("\nNo results returned from crawler")
                
        except Exception as e:
            print(f"\nError crawling falconebiz URL {url}: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
    
    return ""

def read_file(company_name: str, source: str = "thecompanycheck"):
    """
    Read the content of the file for the given company and source.
    Returns the text content or raises an error if the file cannot be read.
    """
    file_path = f"financial_data/{company_name}_{source}_structured.txt"
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        raise
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        raise

def extract_financial_metrics(text: str, company_name: str):
    """
    Extract financial metrics and charge-related details from the text using Gemini API.
    Returns a dictionary with company name, financial year, metrics, and charge details.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

    prompt = """
    You are a financial data extraction assistant. From the provided text, extract the following information for the company described in the text:

    1. Company Name (as a string, use '{company_name}' if not found in text)
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
    prompt = prompt.format(text=text, company_name=company_name)

    try:
        response = model.generate_content(prompt)
        response_text = response.text
        response_text = re.sub(r'```json\n|\n```', '', response_text).strip()
        extracted_data = json.loads(response_text)
        return extracted_data
    except Exception as e:
        print(f"Error processing with Gemini API: {e}")
        return {}

async def main():
    """
    Main function to extract CIN from zaubacorp.com, crawl thecompanycheck.com and falconebiz.com, and extract financial metrics.
    """
    max_results = 1

    # Step 1: Search for zaubacorp.com URL
    zaubacorp_urls = await tavily_search_company(COMPANY_NAME, max_results)
    if not zaubacorp_urls:
        print("No zaubacorp URL found for the company")
        return

    # Step 2: Crawl zaubacorp.com to get CIN
    cin_code = await crawl_zaubacorp_for_cin(zaubacorp_urls[0])
    if not cin_code:
        print("No CIN code extracted from zaubacorp URL")
        return
    print(f"Extracted CIN: {cin_code}")

    # Step 3: Construct URLs
    companycheck_url = construct_companycheck_url(COMPANY_NAME, cin_code)
    print(f"Constructed thecompanycheck URL: {companycheck_url}")
    falconebiz_url = construct_falconebiz_url(COMPANY_NAME, cin_code)
    print(f"Constructed falconebiz URL: {falconebiz_url}")

    # Step 4: Crawl thecompanycheck.com
    companycheck_content = await crawl_companycheck_data(companycheck_url, COMPANY_NAME)
    if not companycheck_content:
        print("No content crawled from thecompanycheck URL")

    # Step 5: Crawl falconebiz.com
    falconebiz_content = await crawl_falconebiz_data(falconebiz_url, COMPANY_NAME)
    if not falconebiz_content:
        print("No content crawled from falconebiz URL")

    # Step 6: Read the saved files and extract financial metrics
    combined_extracted_data = {}
    try:
        if companycheck_content:
            companycheck_text = read_file(COMPANY_NAME, "thecompanycheck")
            companycheck_data = extract_financial_metrics(companycheck_text, COMPANY_NAME)
            combined_extracted_data["thecompanycheck"] = companycheck_data
    except Exception:
        print("Failed to read or process thecompanycheck file")

    try:
        if falconebiz_content:
            falconebiz_text = read_file(COMPANY_NAME, "falconebiz")
            falconebiz_data = extract_financial_metrics(falconebiz_text, COMPANY_NAME)
            combined_extracted_data["falconebiz"] = falconebiz_data
    except Exception:
        print("Failed to read or process falconebiz file")

    # Step 7: Save and print the extracted data
    if combined_extracted_data:
        output_file = f"financial_data/{COMPANY_NAME}_extracted_financial_data.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(combined_extracted_data, f, indent=4, ensure_ascii=False)
        print(f"\nExtracted financial data saved to {output_file}")
        print(json.dumps(combined_extracted_data, indent=4))
    else:
        print("Failed to extract financial metrics and charge details.")

    print("Processing completed successfully")

if __name__ == "__main__":
    asyncio.run(main())