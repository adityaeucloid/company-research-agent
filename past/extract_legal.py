import os
import json
import logging
from typing import Dict, Optional, List
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import openai
import google.generativeai as genai
from pathlib import Path
import importlib.metadata
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extract_legal_report.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure Gemini
load_dotenv()
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

class LegalReportExtractor:
    def __init__(self, api_key: str, company_name: str):
        """
        Initialize the LegalReportExtractor with OpenAI API key and company name.
        
        Args:
            api_key (str): OpenAI API key.
            company_name (str): Name of the company to filter cases for.
        """
        self.api_key = api_key
        self.company_name = company_name
        self.legal_report = {
            "Legal Report": {
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
                "Source File": ""
            }
        }
        # Check OpenAI version and initialize client accordingly
        openai_version = importlib.metadata.version('openai')
        self.is_new_api = openai_version.startswith('1.')
        if self.is_new_api:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            openai.api_key = api_key
            self.client = openai

    def read_file(self, file_path: Path) -> str:
        """
        Read the content of the legal report file.
        
        Args:
            file_path (Path): Path to the file.
        
        Returns:
            str: Content of the file.
        
        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If there's an error reading the file.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except IOError as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise

    def parse_document(self, content: str) -> str:
        """
        Parse the document using BeautifulSoup and extract relevant text.
        
        Args:
            content (str): Raw content of the file.
        
        Returns:
            str: Extracted and cleaned text from relevant sections.
        """
        try:
            soup = BeautifulSoup(content, 'html.parser')
            sections = ['Fact', 'Issue', 'Court\'s Reasoning', 'Conclusion']
            extracted_text = []

            for h2 in soup.find_all('h2'):
                extracted_text.append(h2.get_text(strip=True))

            for tag in soup.find_all(['p', 'pre']):
                title = tag.get('title')
                if title in sections:
                    extracted_text.append(tag.get_text(strip=True))

            return '\n'.join(extracted_text)
        except Exception as e:
            logger.error(f"Error parsing document: {e}")
            raise

    def extract_fields_with_gpt(self, text: str, filename: str) -> Dict:
        """
        Use GPT-4o to extract the specified fields from the text.
        
        Args:
            text (str): Preprocessed text from the document.
            filename (str): Name of the source file.
        
        Returns:
            Dict: Extracted fields in the required JSON format.
        
        Raises:
            Exception: If the API call fails.
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
        {text}
        """
        try:
            if self.is_new_api:
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
                response_text = response.choices[0].message.content
            else:
                response = self.client.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a legal document parser that returns only valid JSON objects."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                response_text = response.choices[0].message.content

            # Clean the response text
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            try:
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
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response: {response_text}")
                raise ValueError(f"Failed to parse GPT response as JSON: {str(e)}")
        except Exception as e:
            logger.error(f"Error extracting fields with GPT-4o: {str(e)}")
            raise

    def process_file(self, file_path: Path) -> Optional[Dict]:
        """
        Process a single legal report file.
        
        Args:
            file_path (Path): Path to the file to process.
        
        Returns:
            Optional[Dict]: Extracted legal report data or None if processing fails.
        """
        try:
            logger.info(f"Processing file: {file_path}")
            content = self.read_file(file_path)
            
            # First check if the case is relevant using Gemini
            if not is_case_relevant(content, self.company_name):
                logger.info(f"Skipping {file_path.name} - not relevant to {self.company_name}")
                return None
                
            logger.info(f"Case is relevant to {self.company_name}")
            parsed_text = self.parse_document(content)
            extracted_data = self.extract_fields_with_gpt(parsed_text, file_path.name)
            logger.info(f"Successfully processed {file_path.name}")
            return extracted_data
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {str(e)}")
            return None

    def process_directory(self, directory_path: str) -> List[Dict]:
        """
        Process all legal report files in a directory.
        
        Args:
            directory_path (str): Path to the directory containing legal report files.
        
        Returns:
            List[Dict]: List of extracted legal report data.
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory_path}")

        legal_reports = []
        for file_path in directory.glob("*.txt"):
            try:
                report = self.process_file(file_path)
                if report:
                    legal_reports.append(report)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue

        return legal_reports

def main():
    """
    Main function to process legal reports and save structured data.
    """
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY not found in .env file")
        raise ValueError("OPENAI_API_KEY not found in .env file")

    company_name = "SL AP PRIVATE LIMITED"
    
    # Initialize extractor
    extractor = LegalReportExtractor(api_key, company_name)
    
    # Process all files in the legal_content directory
    directory_path = "legal_content"
    try:
        legal_reports = extractor.process_directory(directory_path)
        
        if legal_reports:
            # Save all reports to a single JSON file
            output_file = f'legal_content/{company_name}_cases.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(legal_reports, f, indent=2)
            logger.info(f"Saved {len(legal_reports)} legal reports to {output_file}")
            
            # Print summary
            print(f"\nProcessed {len(legal_reports)} relevant files successfully")
            for report in legal_reports:
                print(f"\nFile: {report['Legal Report']['Source File']}")
                print(f"Court: {report['Legal Report']['Court']}")
                print(f"Case Number: {report['Legal Report']['Case Number']}")
                print(f"Status: {report['Legal Report']['Case Status']}")
        else:
            logger.warning("No relevant legal reports were found")
    except Exception as e:
        logger.error(f"Error processing directory: {str(e)}")
        raise

if __name__ == "__main__":
    main()