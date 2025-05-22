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
import google.generativeai as genai
from bs4 import BeautifulSoup

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
    
    parts = url.split("-")
    if len(parts) < 2:
        return False, ""
    
    cin_code = parts[-1]
    if not re.match(r'^[A-Z0-9]{21}$', cin_code):
        return False, ""
    
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
        parts = url.split("-")
        if len(parts) > 1:
            cin_code = parts[-1]
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
        excluded_tags=['header', 'footer', 'form', 'nav', 'script', 'style'],
        cache_mode='BYPASS',
        verbose=True
    )

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)
            
            if isinstance(result, list) and result:
                item = result[0]
                
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
                    for attr_name, attr_value in item.__dict__.items():
                        if isinstance(attr_value, str) and len(attr_value) > 100:
                            content = attr_value
                            break
                
                if content:
                    os.makedirs("crawled_content", exist_ok=True)
                    filename = f"crawled_content/{company_name}_zaubacorp.txt"
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"Saved zaubacorp content to {filename}")
                    
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
    queries = [
        f"site:zaubacorp.com {company_name}",
        f"site:indiafilings.com {company_name}",
        f"site:falconebiz.com/company {company_name} CIN"
    ]
    
    urls = []
    cin_code = ""
    zaubacorp_url = ""
    
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
                url = results["results"][0]["url"]
                if "zaubacorp.com" in url:
                    zaubacorp_url = url
                    print(f"Found zaubacorp URL: {url}")
                    content, extracted_cin = await crawl_zaubacorp_data(url, company_name)
                    if extracted_cin:
                        cin_code = extracted_cin
                        print(f"Extracted CIN {cin_code} from zaubacorp content")
                elif "falconebiz.com" in url:
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
    
    if cin_code:
        falconebiz_url = construct_falconebiz_url(company_name, cin_code)
        if falconebiz_url not in urls:
            urls.append(falconebiz_url)
            print(f"Added falconebiz URL with CIN: {falconebiz_url}")
    
    if zaubacorp_url and zaubacorp_url not in urls:
        urls.append(zaubacorp_url)
        print(f"Added zaubacorp URL: {zaubacorp_url}")
    
    if not urls:
        print("No URLs found from any of the sources")
    else:
        print(f"Found URLs from {len(urls)} sources")
    
    return urls

async def crawl_company_data(urls: list[str], company_name: str, max_pages: int = 10, max_depth: int = 2) -> tuple[str, str]:
    """
    Crawl the provided URLs using advanced crawling techniques and return concatenated markdown content.
    Returns tuple of (falconebiz_content, all_content)
    """
    browser_config = BrowserConfig(
        headless=True,
        browser_type="chromium",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )

    filter_chain = FilterChain([
        DomainFilter(
            allowed_domains=["zaubacorp.com", "indiafilings.com", "falconebiz.com"],
            blocked_domains=["linkedin.com", "facebook.com", "twitter.com"]
        ),
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
        ContentTypeFilter(allowed_types=["text/html"]),
        ContentRelevanceFilter(
            query=f"financial data {company_name}",
            threshold=0.7
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
            max_depth=max_depth,
            include_external=False,
            max_pages=max_pages,
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

    markdown_content = []
    falconebiz_content = ""
    crawl_stats = {
        "total_pages": 0,
        "successful_pages": 0,
        "depth_counts": {},
        "average_score": 0,
        "scores": []
    }

    os.makedirs("crawled_content", exist_ok=True)
    os.makedirs("debug", exist_ok=True)

    print("\nStarting crawl process...")
    print(f"URLs to crawl: {urls}")

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for url in urls:
            try:
                print(f"\nProcessing URL: {url}")
                if "falconebiz.com" in url:
                    is_valid, cin_code = validate_falconebiz_url(url, company_name)
                    if not is_valid:
                        print(f"Skipping invalid falconebiz URL: {url}")
                        continue
                    print(f"Validated falconebiz URL with CIN {cin_code}: {url}")

                result = await crawler.arun(url=url, config=run_config)
                crawl_stats["total_pages"] += 1
                
                parsed_url = urlparse(url)
                website_name = parsed_url.netloc.replace("www.", "").split(".")[0]
                print(f"Website name: {website_name}")
                
                if isinstance(result, list) and result:
                    item = result[0]
                    print(f"\nItem type: {type(item)}")
                    
                    if hasattr(item, 'success'):
                        print(f"Success status: {item.success}")
                    
                    content = None
                    
                    # Try multiple content attributes
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
                        
                        # Save raw content for debugging
                        debug_filename = f"debug/{company_name}_{website_name}_raw.txt"
                        with open(debug_filename, "w", encoding="utf-8") as f:
                            f.write(f"URL: {url}\n")
                            f.write(f"Content length: {len(content)}\n")
                            f.write(f"Content type: {type(content)}\n")
                            f.write("\nContent:\n")
                            f.write(content)
                        
                        # Save processed content
                        filename = f"crawled_content/{company_name}_{website_name}.txt"
                        with open(filename, "w", encoding="utf-8") as f:
                            f.write(content)
                        print(f"Saved content to {filename}")
                        
                        if website_name == "falconebiz":
                            falconebiz_content = content
                            print(f"Stored falconebiz content, length: {len(content)}")
                        markdown_content.append(content)
                    else:
                        print("\nNo content found in the crawled page")
                        
                        # Save debug information
                        debug_filename = f"debug/{company_name}_{website_name}_debug.txt"
                        with open(debug_filename, "w", encoding="utf-8") as f:
                            f.write(f"URL: {url}\n")
                            f.write(f"Item type: {type(item)}\n")
                            f.write(f"Item attributes: {dir(item)}\n")
                            f.write(f"Item dict: {item.__dict__ if hasattr(item, '__dict__') else 'No __dict__'}\n")
                            if hasattr(item, '_results'):
                                f.write("\nResults:\n")
                                for idx, result_item in enumerate(item._results):
                                    f.write(f"\nResult {idx}:\n")
                                    f.write(f"Type: {type(result_item)}\n")
                                    f.write(f"Attributes: {dir(result_item)}\n")
                                    f.write(f"Dict: {result_item.__dict__ if hasattr(result_item, '__dict__') else 'No __dict__'}\n")
                    
                    score = item.metadata.get("score", 0) if hasattr(item, 'metadata') else 0
                    depth = item.metadata.get("depth", 0) if hasattr(item, 'metadata') else 0
                    crawl_stats["scores"].append(score)
                    crawl_stats["depth_counts"][depth] = crawl_stats["depth_counts"].get(depth, 0) + 1
                    
                    print(f"Depth: {depth} | Score: {score:.2f} | {url}")
                else:
                    print(f"Crawl failed for {url}: No valid content extracted")
            except Exception as e:
                print(f"Error crawling {url}: {str(e)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")

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
        return "", ""
    
    print(f"\nFalconebiz content length: {len(falconebiz_content)}")
    print(f"Total content length: {len(''.join(markdown_content))}")
        
    return falconebiz_content, "\n\n".join(markdown_content)

async def extract_company_data(company_name: str, crawled_content: str) -> dict:
    """
    Extract structured company data using Gemini.
    """
    print("\nInitializing Gemini for data extraction...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    print("Gemini model initialized successfully")

    # Save raw content for debugging
    os.makedirs("debug", exist_ok=True)
    debug_file = f"debug/{company_name}_raw_content.txt"
    with open(debug_file, "w", encoding="utf-8") as f:
        f.write(crawled_content)
    print(f"Saved raw content to {debug_file}")
    print(f"Content length: {len(crawled_content)}")

    sys_prompt = """
    You are an expert assistant tasked with extracting structured company data from web content.
    You will receive content from falconebiz.com.
    Your task is to:
    1. Extract all available company information
    2. Ensure accuracy and avoid fabricating data
    3. Focus only on the specified company
    4. IMPORTANT: You must return ONLY a valid JSON object, with no additional text or explanation.
    """
    prompt = f"""
    Extract structured data for the company '{company_name}' from the provided content. The content comes from falconebiz.com.
    
    Return ONLY a valid JSON object matching the following schema, with no additional text or explanation:
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
    - Use "Not Found" for any field where data is unavailable
    - Ensure dates are in DD-MM-YYYY format
    - Extract data only for '{company_name}', ignoring similar companies
    - For Directors, include all listed directors with their DIN/PAN, name, and begin date
    - IMPORTANT: Return ONLY the JSON object, with no additional text or explanation
    
    Content:
    {crawled_content}
    """

    try:
        print("\nSending content to Gemini for extraction...")
        response = model.generate_content([sys_prompt, prompt])
        print("Received response from Gemini")
        
        if response.text:
            print(f"Response length: {len(response.text)}")
            # Save raw response for debugging
            debug_response_file = f"debug/{company_name}_raw_response.txt"
            with open(debug_response_file, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"Saved raw response to {debug_response_file}")
            
            # Try multiple parsing methods
            print("\nAttempting to parse JSON response...")
            try:
                # Method 1: Direct JSON parsing
                print("Method 1: Direct JSON parsing")
                return json.loads(response.text)
            except json.JSONDecodeError as e:
                print(f"Method 1 failed: {str(e)}")
                try:
                    # Method 2: Remove markdown code fences and parse
                    print("Method 2: Remove markdown code fences and parse")
                    cleaned = re.sub(r"```(?:json)?", "", response.text, flags=re.IGNORECASE).strip()
                    return json.loads(cleaned)
                except json.JSONDecodeError as e:
                    print(f"Method 2 failed: {str(e)}")
                    try:
                        # Method 3: Find first { and last } and parse
                        print("Method 3: Find first { and last } and parse")
                        start = response.text.find("{")
                        end = response.text.rfind("}") + 1
                        if start != -1 and end != 0:
                            json_str = response.text[start:end]
                            return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        print(f"Method 3 failed: {str(e)}")
                        try:
                            # Method 4: Use JSONDecoder for more lenient parsing
                            print("Method 4: Use JSONDecoder for more lenient parsing")
                            decoder = json.JSONDecoder()
                            idx = response.text.find("{")
                            if idx != -1:
                                obj, _ = decoder.raw_decode(response.text[idx:])
                                if isinstance(obj, dict):
                                    return obj
                        except Exception as e:
                            print(f"Method 4 failed: {str(e)}")
            
            print("\nAll JSON parsing methods failed. Raw response:")
            print(response.text)
            return {}
        else:
            print("No content extracted from Gemini")
            return {}
    except Exception as e:
        print(f"Error with Gemini extraction: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {}

async def main():
    """
    Main function to run the company data crawler and extractor.
    """
    company_name = "VALUE POINT SYSTEMS PRIVATE LIMITED"
    max_results = 10
    max_pages = 10
    max_depth = 2

    print("\n=== Starting Company Data Extraction Process ===")
    
    # Step 1: Perform Tavily search
    print("\nStep 1: Performing Tavily search...")
    urls = await tavily_search_company(company_name, max_results)
    if not urls:
        print("No relevant URLs found from Tavily search")
        return
    print(f"Found {len(urls)} URLs to process")

    # Step 2: Crawl the URLs
    print("\nStep 2: Crawling URLs...")
    falconebiz_content, all_content = await crawl_company_data(urls, company_name, max_pages, max_depth)
    print(f"Falconebiz content length from crawler: {len(falconebiz_content) if falconebiz_content else 0}")
    
    # Step 3: If falconebiz content is empty, try to read from saved file
    if not falconebiz_content:
        falconebiz_file = f"crawled_content/{company_name}_falconebiz.txt"
        print(f"\nStep 3: Checking for saved falconebiz content in {falconebiz_file}")
        if os.path.exists(falconebiz_file):
            print(f"Found saved falconebiz file, reading content...")
            with open(falconebiz_file, "r", encoding="utf-8") as f:
                falconebiz_content = f.read()
            print(f"Read {len(falconebiz_content)} characters from falconebiz file")
        else:
            print(f"Saved falconebiz file not found at {falconebiz_file}")
    
    if not falconebiz_content:
        print("No falconebiz content available for extraction")
        return

    # Step 4: Extract structured data using falconebiz content
    print("\nStep 4: Extracting data from falconebiz content...")
    print(f"Content length being sent to Gemini: {len(falconebiz_content)}")
    print("First 500 characters of content:")
    print(falconebiz_content[:500])
    
    company_data = await extract_company_data(company_name, falconebiz_content)
    if company_data:
        print(f"\nSuccessfully extracted data for {company_name}:")
        print(json.dumps(company_data, indent=2))
        
        # Save the extracted data
        output_file = f"crawled_content/{company_name}_extracted.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(company_data, f, indent=2)
        print(f"\nSaved extracted data to {output_file}")
    else:
        print("\nFailed to extract data from content")

if __name__ == "__main__":
    asyncio.run(main()) 