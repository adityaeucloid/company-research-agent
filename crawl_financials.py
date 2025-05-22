import asyncio
import os
import re
import logging
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
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('financials_crawler.log')
    ]
)
logger = logging.getLogger(__name__)

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

async def crawl_zaubacorp_for_cin(url: str) -> str:
    """
    Crawl zaubacorp.com URL to extract CIN only.
    """
    logger.info(f"Starting zaubacorp CIN extraction for URL: {url}")
    
    browser_config = BrowserConfig(
        headless=True,
        browser_type="chromium",
        timeout=30000  # 30 seconds timeout
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
        verbose=True,
        timeout=30000  # 30 seconds timeout
    )

    try:
        logger.info("Initializing AsyncWebCrawler")
        async with AsyncWebCrawler(config=browser_config) as crawler:
            logger.info("Starting crawl")
            result = await crawler.arun(url=url, config=run_config)
            logger.info(f"Crawl completed. Result type: {type(result)}")
            
            if isinstance(result, list) and result:
                cin_code = extract_cin_from_url(url)
                logger.info(f"Extracted CIN: {cin_code}")
                return cin_code
            logger.warning(f"Failed to process zaubacorp URL: {url}")
            return ""
    except Exception as e:
        logger.error(f"Error crawling zaubacorp URL {url}: {str(e)}", exc_info=True)
        return ""

async def tavily_search_company(company_name: str, max_results: int = 1) -> list[str]:
    """
    Perform a Tavily search for zaubacorp.com URL of the company.
    """
    logger.info(f"Starting Tavily search for company: {company_name}")
    
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        logger.error("TAVILY_API_KEY not found in environment variables")
        raise ValueError("TAVILY_API_KEY not found in environment variables")
    
    client = TavilyClient(api_key=tavily_api_key)
    query = f"site:zaubacorp.com {company_name}"
    
    try:
        logger.info(f"Executing Tavily search with query: {query}")
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
        logger.error(f"Tavily search failed for query '{query}': {e}", exc_info=True)
        return []

async def crawl_companycheck_data(url: str, company_name: str) -> str:
    """
    Crawl thecompanycheck.com URL and return concatenated markdown content.
    """
    logger.info(f"Starting companycheck crawl for URL: {url}")
    
    browser_config = BrowserConfig(
        headless=False,  # Set to False to handle JavaScript
        browser_type="chromium",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        timeout=30000  # 30 seconds timeout
    )

    filter_chain = FilterChain([
        DomainFilter(
            allowed_domains=["thecompanycheck.com"],
            blocked_domains=[]
        ),
        URLPatternFilter(
            patterns=[url]  # Only allow the exact URL we want to crawl
        ),
        ContentTypeFilter(allowed_types=["text/html"]),
        ContentRelevanceFilter(
            query=f"financial data {company_name}",
            threshold=0.5
        )
    ])

    run_config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=0,
            include_external=False,
            max_pages=1,
            filter_chain=filter_chain
        ),
        excluded_tags=['header', 'footer', 'form', 'nav', 'script', 'style'],
        cache_mode='BYPASS',
        verbose=True,
        timeout=30000,  # 30 seconds timeout
        extraction_strategy=LLMExtractionStrategy(
            extract_metadata=True,
            extract_links=False
        )
    )

    os.makedirs("financial_data", exist_ok=True)
    structured_filename = f"financial_data/{company_name}_thecompanycheck_structured.txt"

    try:
        logger.info("Initializing AsyncWebCrawler")
        async with AsyncWebCrawler(config=browser_config) as crawler:
            logger.info("Starting crawl")
            result = await crawler.arun(url=url, config=run_config)
            logger.info(f"Crawl completed. Result type: {type(result)}")
            
            if isinstance(result, list) and result:
                item = result[0]
                logger.info(f"Processing item. Type: {type(item)}")
                
                if hasattr(item, 'success'):
                    logger.info(f"Success status: {item.success}")
                
                content = None
                
                # Try to get content from _results first
                if hasattr(item, '_results'):
                    logger.info("Checking _results attribute")
                    results = item._results
                    if isinstance(results, list) and results:
                        first_result = results[0]
                        logger.info(f"First result type: {type(first_result)}")
                        
                        content_attrs = [
                            'html',
                            'cleaned_html',
                            'fit_html',
                            'extracted_content',
                            'markdown',
                            'markdown_v2',
                            'fit_markdown'
                        ]
                        
                        for attr in content_attrs:
                            if hasattr(first_result, attr):
                                content = getattr(first_result, attr)
                                logger.info(f"Found content in {attr}")
                                logger.info(f"Content length: {len(content) if content else 0}")
                                if content:
                                    break
                
                # If still no content, try the original item's attributes
                if not content:
                    logger.info("Trying original item attributes")
                    if hasattr(item, 'markdown'):
                        if hasattr(item.markdown, 'fit_markdown'):
                            content = item.markdown.fit_markdown
                            logger.info("Extracted content from fit_markdown")
                        elif hasattr(item.markdown, 'content'):
                            content = item.markdown.content
                            logger.info("Extracted content from markdown.content")
                
                if content:
                    logger.info(f"Final content length: {len(content)}")
                    
                    # Parse HTML and extract structured data
                    soup = BeautifulSoup(content, 'html.parser')
                    extracted_data = []
                    
                    # Get company name
                    company_name_elem = soup.find('h1')
                    if company_name_elem:
                        extracted_data.append(f"Company Name: {company_name_elem.text.strip()}")
                    
                    # Get all tables
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
                    
                    # Get all sections
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
                    
                    # Get company details section
                    company_details = soup.find('div', class_='company-details')
                    if company_details:
                        logger.info("Found company-details section")
                        extracted_data.append("\nCompany Details:")
                        for detail in company_details.find_all(['p', 'div']):
                            text = detail.text.strip()
                            if text and ":" in text and not text.startswith("₹"):
                                extracted_data.append(text)
                    
                    # Save structured data
                    if extracted_data:
                        with open(structured_filename, "w", encoding="utf-8") as f:
                            f.write("\n".join(extracted_data))
                        logger.info(f"Successfully saved structured content to {structured_filename}")
                        return "\n".join(extracted_data)
                    else:
                        logger.warning("No structured data extracted from the page")
                else:
                    logger.warning("No content found in the crawled page")
            else:
                logger.warning("No results returned from crawler")
                
    except Exception as e:
        logger.error(f"Error crawling {url}: {str(e)}", exc_info=True)
        return ""

async def main(company_name: str):
    """
    Main function to extract CIN from zaubacorp.com and crawl thecompanycheck.com.
    """
    logger.info(f"Starting main pipeline for company: {company_name}")
    
    max_results = 1

    try:
        # Step 1: Search for zaubacorp.com URL
        logger.info("Starting zaubacorp URL search")
        zaubacorp_urls = await tavily_search_company(company_name, max_results)
        if not zaubacorp_urls:
            logger.warning("No zaubacorp URL found for the company")
            return

        # Step 2: Crawl zaubacorp.com to get CIN only
        logger.info("Starting CIN extraction from zaubacorp")
        cin_code = await crawl_zaubacorp_for_cin(zaubacorp_urls[0])
        if not cin_code:
            logger.warning("No CIN code extracted from zaubacorp URL")
            return
        logger.info(f"Extracted CIN: {cin_code}")

        # Step 3: Construct thecompanycheck.com URL
        companycheck_url = construct_companycheck_url(company_name, cin_code)
        logger.info(f"Constructed thecompanycheck URL: {companycheck_url}")

        # Step 4: Crawl thecompanycheck.com
        logger.info("Starting companycheck crawl")
        companycheck_content = await crawl_companycheck_data(companycheck_url, company_name)
        if not companycheck_content:
            logger.warning("No content crawled from thecompanycheck URL")
            return

        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}", exc_info=True)

if __name__ == "__main__":
    # For testing purposes
    test_company = "VALUE POINT SYSTEMS PRIVATE LIMITED"
    asyncio.run(main(test_company))