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
import time

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

async def crawl_companycheck_data(url: str, company_name: str) -> str:
    """
    Crawl thecompanycheck.com URL and return concatenated markdown content.
    """
    browser_config = BrowserConfig(
        headless=False,  # Set to False to handle JavaScript
        browser_type="chromium",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )

    # Create a sophisticated filter chain
    filter_chain = FilterChain([
        # Domain boundaries
        DomainFilter(
            allowed_domains=["thecompanycheck.com"],
            blocked_domains=[]
        ),
        # URL patterns to include
        URLPatternFilter(
            patterns=[url]  # Only allow the exact URL we want to crawl
        ),
        # Content type filtering
        ContentTypeFilter(allowed_types=["text/html"]),
        # Content relevance filter
        ContentRelevanceFilter(
            query=f"financial data {company_name}",
            threshold=0.5  # Lower threshold to capture more content
        )
    ])

    # Create a relevance scorer for company-specific content
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
                
                # Try to get content from _results first
                if hasattr(item, '_results'):
                    print("\nChecking _results attribute...")
                    results = item._results
                    if isinstance(results, list) and results:
                        first_result = results[0]
                        print(f"First result type: {type(first_result)}")
                        
                        # Try different content attributes in order of preference
                        content_attrs = [
                            'html',  # Raw HTML
                            'cleaned_html',  # Cleaned HTML
                            'fit_html',  # Fitted HTML
                            'extracted_content',  # Extracted content
                            'markdown',  # Markdown content
                            'markdown_v2',  # Alternative markdown
                            'fit_markdown'  # Fitted markdown
                        ]
                        
                        for attr in content_attrs:
                            if hasattr(first_result, attr):
                                content = getattr(first_result, attr)
                                print(f"\nFound content in {attr}")
                                print(f"Content length: {len(content) if content else 0}")
                                if content:
                                    break
                
                # If still no content, try the original item's attributes
                if not content:
                    print("\nTrying original item attributes...")
                    if hasattr(item, 'markdown'):
                        if hasattr(item.markdown, 'fit_markdown'):
                            content = item.markdown.fit_markdown
                            print("\nExtracted content from fit_markdown")
                            print(f"Content length: {len(content) if content else 0}")
                        elif hasattr(item.markdown, 'content'):
                            content = item.markdown.content
                            print("\nExtracted content from markdown.content")
                            print(f"Content length: {len(content) if content else 0}")
                
                if content:
                    print(f"\nFinal content length: {len(content)}")
                    
                    # Parse HTML and extract structured data
                    soup = BeautifulSoup(content, 'html.parser')
                    extracted_data = []
                    
                    # Get company name
                    company_name_elem = soup.find('h1')
                    if company_name_elem:
                        extracted_data.append(f"Company Name: {company_name_elem.text.strip()}")
                    
                    # Get all tables
                    tables = soup.find_all('table')
                    print(f"\nFound {len(tables)} tables")
                    for table in tables:
                        # Get table title if available
                        table_title = table.find_previous(['h2', 'h3', 'h4', 'div'], class_=re.compile(r'title|heading', re.I))
                        if table_title:
                            extracted_data.append(f"\n{table_title.text.strip()}:")
                        
                        # Extract table data
                        for row in table.find_all('tr'):
                            cells = row.find_all(['td', 'th'])
                            if len(cells) >= 2:
                                label = cells[0].text.strip()
                                value = cells[1].text.strip()
                                if value and value != "GET PRO" and not value.startswith("₹"):
                                    extracted_data.append(f"{label}: {value}")
                    
                    # Get all sections
                    sections = soup.find_all(['div', 'section'], class_=re.compile(r'financial|balance|profit|loss|revenue|assets|liabilities|charges|registered|details|overview', re.I))
                    print(f"\nFound {len(sections)} sections")
                    for section in sections:
                        # Get section title
                        section_title = section.find_previous(['h2', 'h3', 'h4', 'div'], class_=re.compile(r'title|heading', re.I))
                        if section_title:
                            extracted_data.append(f"\n{section_title.text.strip()}:")
                        
                        # Extract text content
                        for p in section.find_all(['p', 'div']):
                            text = p.text.strip()
                            if text and ":" in text and not text.startswith("₹"):
                                extracted_data.append(text)
                    
                    # Get company details section
                    company_details = soup.find('div', class_='company-details')
                    if company_details:
                        print("\nFound company-details section")
                        extracted_data.append("\nCompany Details:")
                        for detail in company_details.find_all(['p', 'div']):
                            text = detail.text.strip()
                            if text and ":" in text and not text.startswith("₹"):
                                extracted_data.append(text)
                    
                    # Save structured data
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

async def main():
    """
    Main function to extract CIN from zaubacorp.com and crawl thecompanycheck.com.
    """
    company_name = "VALUE POINT SYSTEMS PRIVATE LIMITED"
    max_results = 1

    # Step 1: Search for zaubacorp.com URL
    zaubacorp_urls = await tavily_search_company(company_name, max_results)
    if not zaubacorp_urls:
        print("No zaubacorp URL found for the company")
        return

    # Step 2: Crawl zaubacorp.com to get CIN only
    cin_code = await crawl_zaubacorp_for_cin(zaubacorp_urls[0])
    if not cin_code:
        print("No CIN code extracted from zaubacorp URL")
        return
    print(f"Extracted CIN: {cin_code}")

    # Step 3: Construct thecompanycheck.com URL
    companycheck_url = construct_companycheck_url(company_name, cin_code)
    print(f"Constructed thecompanycheck URL: {companycheck_url}")

    # Step 4: Crawl thecompanycheck.com
    companycheck_content = await crawl_companycheck_data(companycheck_url, company_name)
    if not companycheck_content:
        print("No content crawled from thecompanycheck URL")
        return

    print("Crawling completed successfully")

if __name__ == "__main__":
    asyncio.run(main())