import asyncio
import os
import json
import re
from urllib.parse import urlparse
from tavily import TavilyClient
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import (
    ContentRelevanceFilter,
    SEOFilter,
    FilterChain,
    DomainFilter,
    URLPatternFilter,
    ContentTypeFilter
)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.content_filter_strategy import PruningContentFilter
from pydantic import BaseModel
from json import JSONDecoder, JSONDecodeError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
    except JSONDecodeError:
        decoder = JSONDecoder()
        idx = cleaned.find("{")
        if idx != -1:
            obj, _ = decoder.raw_decode(cleaned[idx:])
            if isinstance(obj, dict):
                return obj
        raise ValueError("No valid JSON object found")

def validate_falconebiz_url(url: str, company_name: str) -> tuple[bool, str]:
    """
    Validate falconebiz URL format and extract CIN if present.
    Returns (is_valid, cin_code)
    """
    if not url.startswith("https://www.falconebiz.com/company/"):
        return False, ""
    
    # Extract the last part of the URL after the last hyphen
    parts = url.split("-")
    if len(parts) < 2:
        return False, ""
    
    # Check if the last part matches CIN format (21 alphanumeric characters)
    cin_code = parts[-1]
    if not re.match(r'^[A-Z0-9]{21}$', cin_code):
        return False, ""
    
    # Verify company name is present in URL
    company_slug = company_name.lower().replace(" ", "-")
    url_without_cin = "-".join(parts[:-1])
    if company_slug not in url_without_cin.lower():
        return False, ""
    
    return True, cin_code

def construct_falconebiz_url(company_name: str, cin_code: str = "") -> str:
    """
    Construct a valid falconebiz URL with company name and optional CIN.
    """
    company_slug = company_name.lower().replace(" ", "-")
    if cin_code:
        return f"https://www.falconebiz.com/company/{company_slug}-{cin_code}"
    return f"https://www.falconebiz.com/company/{company_slug}"

def extract_cin_from_url(url: str) -> str:
    """
    Extract CIN code from zaubacorp.com URL.
    """
    if "zaubacorp.com" in url:
        # Extract the last part after the last hyphen
        parts = url.split("-")
        if len(parts) > 1:
            cin_code = parts[-1]
            # Validate if it's a 21-character alphanumeric code
            if re.match(r'^[A-Z0-9]{21}$', cin_code):
                return cin_code
    return ""

async def crawl_zaubacorp_data(url: str, company_name: str) -> tuple[str, str]:
    """
    Crawl zaubacorp.com URL specifically and return both content and CIN.
    """
    browser_config = BrowserConfig(
        headless=True,
        browser_type="chromium"
    )

    # Simpler filter chain for zaubacorp
    filter_chain = FilterChain([
        DomainFilter(
            allowed_domains=["zaubacorp.com"],
            blocked_domains=[]
        ),
        ContentTypeFilter(allowed_types=["text/html"])
    ])

    run_config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=0,  # Don't follow links
            include_external=False,
            max_pages=1,
            filter_chain=filter_chain
        ),
        excluded_tags=['header', 'footer', 'form', 'nav', 'script', 'style'],
        cache_mode='BYPASS',
        verbose=True
    )

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)
            
            if isinstance(result, list) and result:
                item = result[0]
                
                # Try to get content from different possible attributes
                content = ""
                if hasattr(item, 'content'):
                    content = item.content
                elif hasattr(item, 'markdown'):
                    if hasattr(item.markdown, 'fit_markdown'):
                        content = item.markdown.fit_markdown
                    elif hasattr(item.markdown, 'content'):
                        content = item.markdown.content
                elif hasattr(item, 'text'):
                    content = item.text
                elif hasattr(item, 'html'):
                    content = item.html
                elif hasattr(item, 'raw_content'):
                    content = item.raw_content
                elif hasattr(item, 'raw_html'):
                    content = item.raw_html
                
                if not content and hasattr(item, '__dict__'):
                    # Try to get content from any string attribute
                    for attr_name, attr_value in item.__dict__.items():
                        if isinstance(attr_value, str) and len(attr_value) > 100:  # Likely to be content
                            content = attr_value
                            break
                
                if content:
                    # Save content
                    os.makedirs("crawled_content", exist_ok=True)
                    filename = f"crawled_content/{company_name}_zaubacorp.txt"
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"Saved zaubacorp content to {filename}")
                    
                    # Extract CIN
                    cin_code = extract_cin_from_url(url)
                    return content, cin_code
            
            print(f"Failed to extract content from zaubacorp URL: {url}")
            return "", ""
    except Exception as e:
        print(f"Error crawling zaubacorp URL {url}: {str(e)}")
        return "", ""

async def tavily_search_company(company_name: str, max_results: int = 10) -> list[str]:
    """
    Perform a Tavily search for the company and return relevant URLs.
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables")
    
    client = TavilyClient(api_key=tavily_api_key)
    # More specific search queries for each domain
    queries = [
        f"site:zaubacorp.com {company_name}",
        f"site:indiafilings.com {company_name}",
        f"site:falconebiz.com/company {company_name} CIN"  # More specific path
    ]
    
    urls = []
    cin_code = ""
    zaubacorp_url = ""  # Store zaubacorp URL separately
    
    for query in queries:
        try:
            results = client.search(
                query=query,
                max_results=1,  # Get one result per domain
                search_depth="advanced",
                include_domains=["zaubacorp.com", "indiafilings.com", "falconebiz.com"],
                exclude_domains=["linkedin.com", "facebook.com", "twitter.com"]
            )
            if results["results"]:
                url = results["results"][0]["url"]
                if "zaubacorp.com" in url:
                    zaubacorp_url = url
                    print(f"Found zaubacorp URL: {url}")
                    # Crawl zaubacorp first to get content and CIN
                    content, extracted_cin = await crawl_zaubacorp_data(url, company_name)
                    if extracted_cin:
                        cin_code = extracted_cin
                        print(f"Extracted CIN {cin_code} from zaubacorp content")
                elif "falconebiz.com" in url:
                    # Validate falconebiz URL and extract CIN
                    is_valid, extracted_cin = validate_falconebiz_url(url, company_name)
                    if is_valid:
                        cin_code = extracted_cin
                        urls.append(url)
                        print(f"Found valid falconebiz URL with CIN: {url}")
                    else:
                        print(f"Found invalid falconebiz URL: {url}")
                else:
                    urls.append(url)
                    print(f"Found URL for {query}: {url}")
        except Exception as e:
            print(f"Tavily search failed for query '{query}': {e}")
    
    # If we found a CIN (either from zaubacorp or falconebiz), construct the falconebiz URL
    if cin_code:
        falconebiz_url = construct_falconebiz_url(company_name, cin_code)
        if falconebiz_url not in urls:
            urls.append(falconebiz_url)
            print(f"Added falconebiz URL with CIN: {falconebiz_url}")
    
    # Add zaubacorp URL if we found one
    if zaubacorp_url and zaubacorp_url not in urls:
        urls.append(zaubacorp_url)
        print(f"Added zaubacorp URL: {zaubacorp_url}")
    
    if not urls:
        print("No URLs found from any of the sources")
    else:
        print(f"Found URLs from {len(urls)} sources")
    
    return urls

async def crawl_company_data(urls: list[str], company_name: str, max_pages: int = 10, max_depth: int = 2) -> str:
    """
    Crawl the provided URLs using advanced crawling techniques and return concatenated markdown content.
    """
    browser_config = BrowserConfig(
        headless=True,
        browser_type="chromium"
    )

    # Create a sophisticated filter chain
    filter_chain = FilterChain([
        # Domain boundaries - focus on trusted financial and company data sources
        DomainFilter(
            allowed_domains=["zaubacorp.com", "indiafilings.com", "falconebiz.com"],
            blocked_domains=["linkedin.com", "facebook.com", "twitter.com"]
        ),
        # URL patterns to include and exclude
        URLPatternFilter(
            patterns=[
                r"https://.*",
                r"*company*",
                r"*financial*",
                r"*directors*",
                r"*overview*",
                r"*profile*",
                r"*details*"
            ]
        ),
        # Content type filtering
        ContentTypeFilter(allowed_types=["text/html"]),
        # Content relevance filter
        ContentRelevanceFilter(
            query=f"financial data {company_name}",
            threshold=0.7
        )
    ])

    # Create a relevance scorer for company-specific content
    keyword_scorer = KeywordRelevanceScorer(
        keywords=[
            "company", "financial", "directors", "cin", "incorporation",
            "capital", "balance sheet", "annual report", "registration",
            "roc", "status", "address", "email", company_name.lower()
        ],
        weight=0.8  # Increased weight for better relevance scoring
    )

    run_config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=max_depth,
            include_external=False,
            max_pages=max_pages,
            filter_chain=filter_chain,
            url_scorer=keyword_scorer
        ),
        excluded_tags=['header', 'footer', 'form', 'nav', 'script', 'style'],
        cache_mode='BYPASS',
        verbose=True
    )

    markdown_content = []
    crawl_stats = {
        "total_pages": 0,
        "successful_pages": 0,
        "depth_counts": {},
        "average_score": 0,
        "scores": []
    }

    # Create a directory for crawled content if it doesn't exist
    os.makedirs("crawled_content", exist_ok=True)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for url in urls:
            try:
                # For falconebiz, validate URL format
                if "falconebiz.com" in url:
                    is_valid, cin_code = validate_falconebiz_url(url, company_name)
                    if not is_valid:
                        print(f"Skipping invalid falconebiz URL: {url}")
                        continue
                    print(f"Validated falconebiz URL with CIN {cin_code}: {url}")

                result = await crawler.arun(url=url, config=run_config)
                crawl_stats["total_pages"] += 1
                
                # Get website name from URL
                parsed_url = urlparse(url)
                website_name = parsed_url.netloc.replace("www.", "").split(".")[0]
                
                # Handle the result based on its type
                if isinstance(result, list) and result:
                    # Save only the first item
                    item = result[0]
                    if hasattr(item, 'success') and item.success:
                        crawl_stats["successful_pages"] += 1
                        if hasattr(item, 'markdown') and item.markdown:
                            # Save markdown content with specific naming
                            filename = f"crawled_content/{company_name}_{website_name}.txt"
                            with open(filename, "w", encoding="utf-8") as f:
                                f.write(f"Item type: {type(item)}\n")
                                f.write(f"Item dir: {dir(item)}\n")
                                f.write(f"Item dict: {item.__dict__ if hasattr(item, '__dict__') else 'No __dict__'}\n")
                                if hasattr(item.markdown, 'fit_markdown'):
                                    f.write(f"\nFit markdown content:\n{item.markdown.fit_markdown}\n")
                                    markdown_content.append(item.markdown.fit_markdown)
                                if hasattr(item.markdown, 'content'):
                                    f.write(f"\nContent:\n{item.markdown.content}\n")
                                    markdown_content.append(item.markdown.content)
                            
                            # Track metadata
                            score = item.metadata.get("score", 0) if hasattr(item, 'metadata') else 0
                            depth = item.metadata.get("depth", 0) if hasattr(item, 'metadata') else 0
                            crawl_stats["scores"].append(score)
                            crawl_stats["depth_counts"][depth] = crawl_stats["depth_counts"].get(depth, 0) + 1
                            
                            print(f"Depth: {depth} | Score: {score:.2f} | {url}")
                else:
                    print(f"Crawl failed for {url}: No valid content extracted")
            except Exception as e:
                print(f"Error crawling {url}: {str(e)}")

    # Print crawl statistics
    if crawl_stats["total_pages"] > 0:
        print("\nCrawl Statistics:")
        print(f"Total pages attempted: {crawl_stats['total_pages']}")
        print(f"Successfully crawled: {crawl_stats['successful_pages']}")
        if crawl_stats["scores"]:
            print(f"Average relevance score: {sum(crawl_stats['scores']) / len(crawl_stats['scores']):.2f}")
        print("\nPages crawled by depth:")
        for depth, count in sorted(crawl_stats["depth_counts"].items()):
            print(f"  Depth {depth}: {count} pages")

    if not markdown_content:
        print("No content crawled from provided URLs")
        return ""
        
    return "\n\n".join(markdown_content)

async def main(company_name: str):
    """
    Main function to run the company data crawler.
    """
    max_results = 10
    max_pages = 10
    max_depth = 2

    # Perform Tavily search
    urls = await tavily_search_company(company_name, max_results)
    if not urls:
        print("No relevant URLs found from Tavily search")
        return

    # Crawl the URLs
    crawled_content = await crawl_company_data(urls, company_name, max_pages, max_depth)
    if not crawled_content:
        print("No content crawled from provided URLs")
        return

if __name__ == "__main__":
    # For testing purposes
    test_company = "VALUE POINT SYSTEMS PRIVATE LIMITED"
    asyncio.run(main(test_company))