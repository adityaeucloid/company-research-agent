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
import logging
from typing import Optional, Dict, Any, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
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
        logging.FileHandler(LOGS_DIR / 'financial_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class FinancialDataExtractor:
    def __init__(self, company_name: str):
        """Initialize the extractor with company name."""
        self.company_name = company_name.upper()
        self.company_dir = FINANCIAL_DATA_DIR / company_name.replace(" ", "_").lower()
        self.company_dir.mkdir(exist_ok=True)
        self.validate_environment_variables()
        
    def validate_environment_variables(self) -> bool:
        """Validate that all required environment variables are present."""
        required_vars = ["TAVILY_API_KEY", "GEMINI_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            return False
        return True

    def extract_cin_from_url(self, url: str) -> Optional[str]:
        """Extract CIN code from zaubacorp.com URL with validation."""
        try:
            if not url or "zaubacorp.com" not in url:
                logger.warning(f"Invalid URL format: {url}")
                return None
                
            parts = url.split("-")
            if len(parts) <= 1:
                logger.warning(f"URL does not contain expected CIN format: {url}")
                return None
                
            cin_code = parts[-1]
            if not re.match(r'^[A-Z0-9]{21}$', cin_code):
                logger.warning(f"Invalid CIN format: {cin_code}")
                return None
                
            return cin_code
        except Exception as e:
            logger.error(f"Error extracting CIN from URL: {str(e)}")
            return None

    def construct_companycheck_url(self, cin_code: str) -> Optional[str]:
        """Construct a valid thecompanycheck.com URL with validation."""
        try:
            if not self.company_name or not cin_code:
                logger.warning("Missing company name or CIN code")
                return None
                
            company_slug = self.company_name.lower().replace(" ", "-")
            url = f"https://www.thecompanycheck.com/company/{company_slug}/{cin_code}"
            
            # Validate URL format
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc, parsed.path]):
                logger.warning(f"Invalid URL constructed: {url}")
                return None
                
            return url
        except Exception as e:
            logger.error(f"Error constructing companycheck URL: {str(e)}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def tavily_search_company(self, max_results: int = 1) -> list[str]:
        """Perform a Tavily search with retry logic."""
        if not self.validate_environment_variables():
            return []
            
        try:
            client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            query = f"site:zaubacorp.com {self.company_name}"
            
            results = client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_domains=["zaubacorp.com"],
                exclude_domains=["linkedin.com", "facebook.com", "twitter.com"]
            )
            
            urls = [result["url"] for result in results["results"]]
            if urls:
                logger.info(f"Found zaubacorp URL: {urls[0]}")
            else:
                logger.warning("No zaubacorp URL found")
            return urls
        except Exception as e:
            logger.error(f"Tavily search failed: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def crawl_companycheck_data(self, url: str) -> str:
        """Crawl thecompanycheck.com URL with retry logic and improved error handling."""
        if not url:
            logger.error("Invalid URL provided")
            return ""
            
        try:
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
                    query=f"financial data {self.company_name}",
                    threshold=0.5
                )
            ])

            keyword_scorer = KeywordRelevanceScorer(
                keywords=[
                    "company", "financial", "directors", "cin", "incorporation",
                    "capital", "balance sheet", "annual report", "registration",
                    "roc", "status", "address", "email", self.company_name.lower()
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

            structured_filename = self.company_dir / "thecompanycheck_structured.txt"
            raw_filename = self.company_dir / "thecompanycheck_raw.html"

            async with AsyncWebCrawler(config=browser_config) as crawler:
                logger.info(f"Starting crawl for URL: {url}")
                result = await crawler.arun(url=url, config=run_config)
                
                if not result or not isinstance(result, list) or not result:
                    logger.error("No results returned from crawler")
                    return ""
                    
                item = result[0]
                if not hasattr(item, 'success') or not item.success:
                    logger.error("Crawl was not successful")
                    return ""
                    
                content = None
                content_attrs = [
                    'html', 'cleaned_html', 'fit_html', 'extracted_content',
                    'markdown', 'markdown_v2', 'fit_markdown'
                ]
                
                for attr in content_attrs:
                    if hasattr(item, '_results') and isinstance(item._results, list) and item._results:
                        if hasattr(item._results[0], attr):
                            content = getattr(item._results[0], attr)
                            logger.info(f"Found content in {attr}")
                            if content:
                                break
                
                if not content:
                    logger.error("No content found in crawled page")
                    return ""
                    
                # Save raw content
                with open(raw_filename, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.info(f"Saved raw content to {raw_filename}")
                
                # Extract structured data
                soup = BeautifulSoup(content, 'html.parser')
                extracted_data = []
                
                # Extract company name
                company_name_elem = soup.find('h1')
                if company_name_elem:
                    extracted_data.append(f"Company Name: {company_name_elem.text.strip()}")
                
                # Extract table data
                tables = soup.find_all('table')
                logger.info(f"Found {len(tables)} tables")
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
                
                # Extract section data
                sections = soup.find_all(['div', 'section'], class_=re.compile(r'financial|balance|profit|loss|revenue|assets|liabilities|charges|registered|details|overview', re.I))
                logger.info(f"Found {len(sections)} sections")
                for section in sections:
                    section_title = section.find_previous(['h2', 'h3', 'h4', 'div'], class_=re.compile(r'title|heading', re.I))
                    if section_title:
                        extracted_data.append(f"\n{section_title.text.strip()}:")
                    
                    for p in section.find_all(['p', 'div']):
                        text = p.text.strip()
                        if text and ":" in text and not text.startswith("₹"):
                            extracted_data.append(text)
                
                # Extract company details
                company_details = soup.find('div', class_='company-details')
                if company_details:
                    logger.info("Found company-details section")
                    extracted_data.append("\nCompany Details:")
                    for detail in company_details.find_all(['p', 'div']):
                        text = detail.text.strip()
                        if text and ":" in text and not text.startswith("₹"):
                            extracted_data.append(text)
                
                if extracted_data:
                    with open(structured_filename, "w", encoding="utf-8") as f:
                        f.write("\n".join(extracted_data))
                    logger.info(f"Successfully saved structured content to {structured_filename}")
                    return "\n".join(extracted_data)
                else:
                    logger.warning("No structured data extracted from the page")
                    return ""
                    
        except Exception as e:
            logger.error(f"Error crawling {url}: {str(e)}")
            raise

    def read_file(self) -> str:
        """Read the content of the file with improved error handling."""
        file_path = self.company_dir / "thecompanycheck_structured.txt"
        try:
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return ""
                
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                if not content:
                    logger.warning(f"Empty file: {file_path}")
                return content
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return ""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def extract_financial_metrics(self, text: str) -> Dict[str, Any]:
        """Extract financial metrics with retry logic and improved error handling."""
        if not self.validate_environment_variables():
            return {}
            
        if not text:
            logger.error("No text provided for extraction")
            return {}
            
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
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
            """
            prompt = prompt.format(text=text, company_name=self.company_name)

            response = model.generate_content(prompt)
            response_text = response.text
            response_text = re.sub(r'```json\n|\n```', '', response_text).strip()
            
            try:
                extracted_data = json.loads(response_text)
                if not extracted_data:
                    logger.warning("Empty data extracted from Gemini API")
                    return {}
                    
                # Validate required fields
                required_fields = ['company_name', 'financial_year', 'financial_metrics_YoY_growth', 'charge_details_in_cr']
                missing_fields = [field for field in required_fields if field not in extracted_data]
                if missing_fields:
                    logger.warning(f"Missing required fields in extracted data: {missing_fields}")
                    return {}
                    
                return extracted_data
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {str(e)}")
                return {}
                
        except Exception as e:
            logger.error(f"Error processing with Gemini API: {str(e)}")
            raise

    async def process_company_data(self) -> Tuple[Dict[str, Any], str]:
        """Process company data and return financial data and any error message."""
        try:
            # Step 1: Search for zaubacorp.com URL
            zaubacorp_urls = await self.tavily_search_company()
            if not zaubacorp_urls:
                return {}, "No zaubacorp URL found for the company"

            # Step 2: Extract CIN from zaubacorp URL
            cin_code = self.extract_cin_from_url(zaubacorp_urls[0])
            if not cin_code:
                return {}, "No CIN code extracted from zaubacorp URL"
            logger.info(f"Extracted CIN: {cin_code}")

            # Step 3: Construct thecompanycheck.com URL
            companycheck_url = self.construct_companycheck_url(cin_code)
            if not companycheck_url:
                return {}, "Failed to construct companycheck URL"
            logger.info(f"Constructed thecompanycheck URL: {companycheck_url}")

            # Step 4: Crawl thecompanycheck.com
            companycheck_content = await self.crawl_companycheck_data(companycheck_url)
            if not companycheck_content:
                return {}, "No content crawled from thecompanycheck URL"

            # Step 5: Read the saved file
            scraped_text = self.read_file()
            if not scraped_text:
                return {}, "Failed to read the scraped file"

            # Step 6: Extract metrics and charge details
            extracted_data = self.extract_financial_metrics(scraped_text)
            if not extracted_data:
                return {}, "Failed to extract financial metrics and charge details"

            # Step 7: Save the extracted data
            output_file = self.company_dir / "extracted_financial_data.json"
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(extracted_data, f, indent=4, ensure_ascii=False)
                logger.info(f"Extracted financial data saved to {output_file}")
                return extracted_data, ""
            except Exception as e:
                logger.error(f"Error saving extracted data: {str(e)}")
                return {}, f"Error saving extracted data: {str(e)}"

        except Exception as e:
            error_msg = f"Unexpected error in processing: {str(e)}"
            logger.error(error_msg)
            return {}, error_msg

async def main(company_name: str) -> Tuple[Dict[str, Any], str]:
    """Main function to process company data."""
    extractor = FinancialDataExtractor(company_name)
    return await extractor.process_company_data()

if __name__ == "__main__":
    # For testing purposes
    asyncio.run(main("VALUE POINT SYSTEMS PRIVATE LIMITED"))