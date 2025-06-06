import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, quote
import time
import os
import json
import logging
from typing import Dict, Optional, List
from dotenv import load_dotenv
import openai
import google.generativeai as genai
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
import asyncio
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

def is_case_relevant(content: str, company_name: str) -> bool:
    """
    Use Gemini to determine if the case is specifically about the company.
    More lenient classification that includes cases where company appears as respondent/petitioner.
    """
    try:
        # First do a basic text check
        content_lower = content.lower()
        company_name_lower = company_name.lower()
        
        # Check for exact match
        if company_name_lower in content_lower:
            logger.info(f"Found exact match of company name")
            return True
            
        # Check for common variations
        variations = [
            company_name_lower.replace("private limited", "pvt ltd"),
            company_name_lower.replace("private limited", "pvt. ltd."),
            company_name_lower.replace("private limited", "private ltd"),
            company_name_lower.replace("private limited", "pvt. limited"),
            company_name_lower.replace("private limited", "private ltd."),
            # Add company name without "private limited"
            company_name_lower.replace(" private limited", ""),
            company_name_lower.replace(" pvt ltd", ""),
            company_name_lower.replace(" pvt. ltd.", ""),
        ]
        
        for variation in variations:
            if variation in content_lower:
                logger.info(f"Found variation: {variation}")
                return True

        # If basic check fails, use Gemini for more nuanced analysis
        prompt = f"""
        Analyze the following legal case text and determine if it is specifically about {company_name}.
        
        IMPORTANT INSTRUCTIONS:
        1. Be EXTREMELY lenient in classification
        2. If there's ANY mention of the company or its variations, mark as relevant
        3. Consider the case relevant if:
           - Company name appears in any form
           - Company is mentioned as a party
           - Company is referenced in the judgment
           - Company is mentioned in any context
           - Company's actions or rights are discussed
           - Company is involved in any way
        
        Return your answer in this exact format:
        {{
            "is_relevant": true/false,
            "reason": "Detailed explanation of why this case is or isn't relevant",
            "company_role": "Role of the company in the case (e.g., petitioner, respondent, subject)",
            "mentions": "List of specific mentions or references to the company in the text, including any variations of the name found"
        }}

        Text to analyze:
        {content[:4000]}  # Limit content to avoid token limits
        """

        response = gemini_model.generate_content(prompt)
        response_text = response.text

        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            result = json.loads(json_match.group(0))
            logger.info(f"\nCase relevance analysis:")
            logger.info(f"Reason: {result['reason']}")
            logger.info(f"Company role: {result['company_role']}")
            logger.info(f"Mentions: {result['mentions']}")
            return result['is_relevant']
        
        logger.warning("Failed to extract JSON from Gemini response")
        # If we can't get a clear answer from Gemini, be lenient and mark as relevant
        return True
    except Exception as e:
        logger.error(f"Error checking case relevance: {str(e)}")
        # If there's an error, be lenient and mark as relevant
        return True

class LegalCaseExtractor:
    def __init__(self, api_key: str, company_name: str):
        """
        Initialize the LegalCaseExtractor with OpenAI API key and company name.
        """
        self.api_key = api_key
        self.company_name = company_name
        self.client = openai.OpenAI(api_key=api_key)
        
    def extract_case_data(self, content: str, filename: str) -> Dict:
        """
        Use GPT-4o to extract structured data from the case content.
        """
        prompt = f"""
        You are an expert in extracting information from legal documents. Extract the following fields from the provided legal text and return ONLY a valid JSON object with this exact structure:

        {{
            "Legal Report": {{
                "Court Type": "",
                "Court": "",
                "Case Subject": "",
                "Case Type": "",
                "Case Number": "",
                "Appellant": "",
                "Respondent": "",
                "Disposed Date": "",
                "Case Status": "",
                "Remarks": "",
                "Source File": "{filename}"
            }}
        }}

        IMPORTANT:
        1. Return ONLY the JSON object, no other text or explanation
        2. Ensure all fields are present in the output
        3. Use empty string ("") for any field where information is not found
        4. Do not include any markdown formatting or backticks
        5. The response must be a valid JSON object that can be parsed by json.loads()

        Field Guidelines:
        - Court Type: Type of court (e.g., Appellate Tribunal, High Court)
        - Court: Full court name with location (e.g., Income Tax Appellate Tribunal - Bangalore)
        - Case Subject: Main subject (e.g., Income Tax Dispute)
        - Case Type: Type of case (e.g., Income Tax Appeal)
        - Case Number: Complete case number (e.g., ITA.376/Bang/2017)
        - Appellant: Full name of appellant
        - Respondent: Full name of respondent
        - Disposed Date: Date in DD-MM-YYYY format
        - Case Status: Current status (e.g., Allowed, Dismissed, Pending)
        - Remarks: Brief summary of the case outcome
        - Source File: Name of the source file (already provided)

        Text to analyze:
        {content}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a legal document parser that returns only valid JSON objects."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Clean the response text
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            extracted_data = json.loads(response_text)
            
            # Validate the structure
            if "Legal Report" not in extracted_data:
                raise ValueError("Response missing 'Legal Report' key")
                
            required_fields = [
                "Court Type", "Court", "Case Subject", "Case Type",
                "Case Number", "Appellant", "Respondent", "Disposed Date",
                "Case Status", "Remarks", "Source File"
            ]
            
            for field in required_fields:
                if field not in extracted_data["Legal Report"]:
                    if field == "Source File":
                        extracted_data["Legal Report"][field] = filename
                    else:
                        extracted_data["Legal Report"][field] = ""
                        
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting case data: {str(e)}")
            raise

async def search_kanoon_cases(company_name: str, max_results: int = 10) -> list:
    """
    Search IndianKanoon for legal cases related to a company and return the URLs.
    """
    # Construct the search URL
    encoded_company = quote(company_name)
    search_url = f"https://indiankanoon.org/search/?formInput={encoded_company}"
    
    # Headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Make the request
        print(f"Searching for cases related to: {company_name}")
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all case links
        case_urls = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Only include URLs that are actual case documents
            if '/doc/' in href and '/undefined' not in href:
                full_url = f"https://indiankanoon.org{href}"
                if full_url not in case_urls:  # Avoid duplicates
                    case_urls.append(full_url)
                    print(f"Found case: {full_url}")
                    
                    # Break if we have enough results
                    if len(case_urls) >= max_results:
                        break
        
        print(f"\nFound {len(case_urls)} case URLs")
        return case_urls[:max_results]
        
    except requests.RequestException as e:
        print(f"Error making request: {str(e)}")
        return []
    except Exception as e:
        print(f"Error processing results: {str(e)}")
        return []

async def crawl_single_url(url: str, company_name: str, crawler: AsyncWebCrawler) -> Optional[str]:
    """
    Crawl a single URL and return its content.
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
            threshold=0.2
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
            max_pages=1,
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
                    
                    if hasattr(item, '_results'):
                        results = item._results
                        if results and len(results) > 0:
                            first_result = results[0]
                            
                            # Try different content extraction methods
                            if hasattr(first_result, 'content'):
                                content = first_result.content
                            elif hasattr(first_result, 'text'):
                                content = first_result.text
                            elif hasattr(first_result, 'raw_content'):
                                content = first_result.raw_content
                            elif hasattr(first_result, 'raw_html'):
                                content = first_result.raw_html
                            
                            # If still no content, try to get any string attribute
                            if not content:
                                for attr in dir(first_result):
                                    if not attr.startswith('_'):
                                        attr_value = getattr(first_result, attr)
                                        if isinstance(attr_value, str) and len(attr_value) > 100:
                                            content = attr_value
                                            break
                    
                    if content:
                        return content
                    
        return None
    except Exception as e:
        logger.error(f"Error crawling {url}: {str(e)}")
        return None

async def process_cases(company_name: str, urls: List[str], extractor: LegalCaseExtractor) -> List[Dict]:
    """
    Process multiple case URLs and extract structured data.
    """
    os.makedirs("legal_content", exist_ok=True)
    legal_reports = []
    
    async with AsyncWebCrawler(config=BrowserConfig(headless=True, browser_type="chromium")) as crawler:
        for url in urls:
            print(f"\nProcessing URL: {url}")
            
            # Crawl the URL
            content = await crawl_single_url(url, company_name, crawler)
            if not content:
                print(f"Failed to get content from {url}")
                continue
            
            # Check if case is relevant
            if not is_case_relevant(content, company_name):
                print(f"Skipping {url} - not relevant to {company_name}")
                continue
                
            # Save raw content
            parsed_url = urlparse(url)
            filename = f"legal_content/{company_name}_{parsed_url.path.replace('/', '_')}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Saved content to {filename}")
            
            # Extract structured data
            try:
                case_data = extractor.extract_case_data(content, filename)
                legal_reports.append(case_data)
                print(f"Successfully extracted data from {url}")
            except Exception as e:
                logger.error(f"Error extracting data from {url}: {str(e)}")
                continue
    
    return legal_reports

async def main():
    """
    Main function to run the legal case crawler and extractor.
    """
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Company name to search for
    company_name = "VALUE POINT SYSTEMS PRIVATE LIMITED"
    
    # Initialize extractor
    extractor = LegalCaseExtractor(api_key, company_name)
    
    # Search for cases
    urls = await search_kanoon_cases(company_name)
    if not urls:
        print("No legal case URLs found")
        return
    
    print(f"\nFound {len(urls)} URLs to process")
    
    # Process cases and extract data
    legal_reports = await process_cases(company_name, urls, extractor)
    
    if legal_reports:
        # Save all reports to a single JSON file
        output_file = f'legal_content/{company_name}_cases.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(legal_reports, f, indent=2)
        print(f"\nSaved {len(legal_reports)} legal reports to {output_file}")
        
        # Print summary
        print("\nProcessed cases summary:")
        for report in legal_reports:
            print(f"\nFile: {report['Legal Report']['Source File']}")
            print(f"Court: {report['Legal Report']['Court']}")
            print(f"Case Number: {report['Legal Report']['Case Number']}")
            print(f"Status: {report['Legal Report']['Case Status']}")
    else:
        print("No legal reports were extracted")

if __name__ == "__main__":
    asyncio.run(main())
