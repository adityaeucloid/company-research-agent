import asyncio
import os
from urllib.parse import urlparse, quote
from tavily import TavilyClient
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import (
    ContentRelevanceFilter,
    FilterChain,
    DomainFilter,
    URLPatternFilter,
    ContentTypeFilter
)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from dotenv import load_dotenv
from typing import List
import aiohttp
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

async def search_indiankanoon_direct(company_name: str) -> List[str]:
    """
    Search directly on IndianKanoon website for legal cases.
    """
    urls = set()
    try:
        # Construct search URL
        encoded_company = quote(company_name)
        search_url = f"https://indiankanoon.org/search/?formInput={encoded_company}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find all case links
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if '/doc/' in href and '/undefined' not in href:
                            full_url = f"https://indiankanoon.org{href}"
                            urls.add(full_url)
                            print(f"Found direct IndianKanoon case: {full_url}")
        
        print(f"Found {len(urls)} cases from direct IndianKanoon search")
        return list(urls)
    except Exception as e:
        print(f"Error in direct IndianKanoon search: {str(e)}")
        return []

async def search_legal_cases(company_name: str) -> List[str]:
    """
    Search for legal cases using both Tavily and direct IndianKanoon search.
    """
    try:
        # First, do direct IndianKanoon search
        indiankanoon_urls = await search_indiankanoon_direct(company_name)
        
        # Then do Tavily search
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")
        
        client = TavilyClient(api_key=tavily_api_key)
        
        # Construct multiple search queries to find more cases
        search_queries = [
            f'"{company_name}" site:indiankanoon.org',  # Basic company search
            f'"{company_name}" site:indiankanoon.org (case OR judgment OR petition OR appeal)',  # Legal terms
            f'"{company_name}" site:indiankanoon.org (tax OR income OR gst OR vat)',  # Tax related
            f'"{company_name}" site:indiankanoon.org (writ OR order OR judgment)',  # Court orders
            f'"{company_name}" site:indiankanoon.org (supreme court OR high court)',  # Court specific
            f'"{company_name}" site:indiankanoon.org (income tax OR direct tax)',  # Tax specific
            f'"{company_name}" site:indiankanoon.org (corporate OR company)',  # Corporate specific
            f'"{company_name}" site:indiankanoon.org (commissioner OR additional commissioner OR assistant commissioner)', 
        ]
        
        tavily_urls = set()
        for query in search_queries:
            # Configure search parameters for better legal case results
            search_params = {
                "query": query,
                "search_depth": "advanced",  # Use advanced search for better results
                "include_domains": ["indiankanoon.org"],  # Focus on IndianKanoon
                "exclude_domains": ["indiankanoon.org/doc/undefined"],  # Exclude invalid documents
                "max_results": 10,  # Get top 10 results per query
                "include_answer": True,  # Include answer for better context
                "include_raw_content": True,  # Include raw content for better relevance
                "sort_by": "relevance"  # Sort by relevance
            }
            
            print(f"\nSearching with query: {query}")
            response = client.search(**search_params)
            
            # Extract and filter URLs
            for result in response.get("results", []):
                url = result.get("url", "")
                # Only include URLs that are actual case documents
                if "indiankanoon.org/doc/" in url and "/undefined" not in url:
                    tavily_urls.add(url)
                    print(f"Found legal case URL: {url}")
        
        # Combine URLs from both sources
        all_urls = list(set(indiankanoon_urls + list(tavily_urls)))
        print(f"\nFound {len(all_urls)} unique case URLs in total")
        print(f"- {len(indiankanoon_urls)} from direct IndianKanoon search")
        print(f"- {len(tavily_urls)} from Tavily search")
        
        return all_urls
    except Exception as e:
        print(f"Error searching for legal cases: {str(e)}")
        return []

async def crawl_single_url(url: str, company_name: str, crawler: AsyncWebCrawler) -> bool:
    """
    Crawl a single URL and save its content.
    """
    browser_config = BrowserConfig(
        headless=True,
        browser_type="chromium"
    )

    # Create filter chain for legal content
    filter_chain = FilterChain([
        DomainFilter(
            allowed_domains=["indiankanoon.org"],
            blocked_domains=[]
        ),
        URLPatternFilter(
            patterns=[
                r"https://.*",
                r"*doc*"
            ]
        ),
        ContentTypeFilter(allowed_types=["text/html"]),
        ContentRelevanceFilter(
            query=f"legal case {company_name}",
            threshold=0.2  # Increased threshold for better relevance
        )
    ])

    # Create relevance scorer for legal content
    keyword_scorer = KeywordRelevanceScorer(
        keywords=[
            "court", "tribunal", "judgment", "order", "case", "appeal",
            "petition", "respondent", "appellant", "disposed", "dismissed",
            company_name.lower()
        ],
        weight=0.8
    )

    run_config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=1,
            include_external=False,
            max_pages=1,  # Only crawl the main page
            filter_chain=filter_chain,
            url_scorer=keyword_scorer
        ),
        excluded_tags=['header', 'footer', 'form', 'nav', 'script', 'style'],
        cache_mode='BYPASS',
        verbose=True
    )

    try:
        result = await crawler.arun(url=url, config=run_config)
        
        if isinstance(result, list) and result:
            for item in result:
                if hasattr(item, 'success') and item.success:
                    content = None
                    
                    # Debug: Print available attributes
                    print(f"\nAvailable attributes for item: {dir(item)}")
                    
                    # Access the _results attribute
                    if hasattr(item, '_results'):
                        results = item._results
                        print(f"Found _results attribute with {len(results)} items")
                        
                        # Try to get content from the first result
                        if results and len(results) > 0:
                            first_result = results[0]
                            print(f"First result attributes: {dir(first_result)}")
                            
                            # Try different content extraction methods
                            if hasattr(first_result, 'content'):
                                content = first_result.content
                                print("Using first_result.content")
                            elif hasattr(first_result, 'text'):
                                content = first_result.text
                                print("Using first_result.text")
                            elif hasattr(first_result, 'raw_content'):
                                content = first_result.raw_content
                                print("Using first_result.raw_content")
                            elif hasattr(first_result, 'raw_html'):
                                content = first_result.raw_html
                                print("Using first_result.raw_html")
                            
                            # If still no content, try to get any string attribute
                            if not content:
                                for attr in dir(first_result):
                                    if not attr.startswith('_'):
                                        attr_value = getattr(first_result, attr)
                                        if isinstance(attr_value, str) and len(attr_value) > 100:
                                            content = attr_value
                                            print(f"Using first_result.{attr}")
                                            break
                    
                    if content:
                        # Save raw content
                        parsed_url = urlparse(url)
                        filename = f"legal_content/{company_name}_{parsed_url.path.replace('/', '_')}.txt"
                        with open(filename, "w", encoding="utf-8") as f:
                            f.write(content)
                        print(f"Saved content to {filename}")
                        return True
                    else:
                        print("No content found in any attribute")
        else:
            print("No results returned from crawler")
        return False
    except Exception as e:
        print(f"Error crawling {url}: {str(e)}")
        return False

async def crawl_legal_cases(urls: list[str], company_name: str) -> None:
    """
    Crawl the provided URLs and save content to files.
    """
    os.makedirs("legal_content", exist_ok=True)
    
    async with AsyncWebCrawler(config=BrowserConfig(headless=True, browser_type="chromium")) as crawler:
        for url in urls:
            print(f"\nProcessing URL: {url}")
            success = await crawl_single_url(url, company_name, crawler)
            if success:
                print(f"Successfully processed {url}")
            else:
                print(f"Failed to process {url}")

async def main():
    """
    Main function to run the legal case crawler.
    """
    company_name = "SL AP PRIVATE LIMITED"

    # Search for legal cases
    urls = await search_legal_cases(company_name)
    if not urls:
        print("No legal case URLs found")
        return

    print(f"\nFound {len(urls)} URLs to process")
    
    # Crawl and save content
    await crawl_legal_cases(urls, company_name)
    print("\nCrawling completed. Check legal_content folder for results.")

if __name__ == "__main__":
    asyncio.run(main()) 