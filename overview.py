from openai import OpenAI
from dotenv import load_dotenv
import os
import re
import json
from json import JSONDecoder, JSONDecodeError

load_dotenv()

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def parse_model_json(model_output: str) -> dict:
    """
    Extract and parse the first valid JSON object found in `model_output`,
    removing any Markdown code fences (``` or ```json```) beforehand.
    """
    # 1) Remove all triple-backtick fences (``` and ```json)
    no_fences = re.sub(r"```(?:json)?", "", model_output, flags=re.IGNORECASE)
    cleaned = no_fences.strip()

    # 2) Use JSONDecoder.raw_decode to find the first valid JSON object
    decoder = JSONDecoder()
    idx = 0
    length = len(cleaned)
    while idx < length:
        # Find the next opening brace
        brace_pos = cleaned.find("{", idx)
        if brace_pos == -1:
            break
        try:
            # Attempt to decode JSON starting here
            obj, end = decoder.raw_decode(cleaned[brace_pos:])
            return obj
        except JSONDecodeError:
            # Move one character forward and retry
            idx = brace_pos + 1
            continue

    # If we get here, no JSON object was successfully decoded
    raise ValueError("No valid JSON object found")

def get_overview_report(company_name, client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))) -> dict:
    """
    Get the overview report of a company using OpenAI's web search capabilities.
    """
    prompt = f"""
    You are a helpful assistant with live web search capabilities, you will be provided with a company name and you will search the web for all the information related to the company with regrards to the following aspects mentioned as JSON keys:
    ```json
    {{
    "CIN Number": "...", (CIN is the Corporate Identification Number)
    "Name": "...", (The official name of the company)
    "Listed on Stock Exchange": "...", (Yes/No)
    "Company Status (for eFiling)": "...", (Active/Inactive)
    "PAN": "...", (Permanent Account Number)
    "Date of Incorporation": "...", (Date when the company was incorporated)
    "LEI Number": "...", (Legal Entity Identifier)
    "LEI Expiry Date": "...", (Expiry date of the LEI)
    "RoC Code": "...", (Registration of Company)
    "email id": "...", (Official email address of the company)
    "Sector": "...", (Sector in which the company operates)
    "Industry": "...", (Industry in which the company operates)
    "website": "...", (Official website of the company)
    "Address": "...", (Official address of the company)
    "State": "...", (State in which the company is registered)
    "Authorised Share Capital": "...",
    "Paid-up Share Capital": "...",
    "Date of Last Filed Balance Sheet": "...",
    "Current Directors & Key Managerial Personnel": [
        {{
        "DIN": "...",
        "Director Name": "...",
        "Designation": "...",
        "Appointment Date": "..."
        }},
        // ... more directors
    ]
    }}
    ```
    You will search the web for the above information and return the result in JSON format. The JSON keys should be exactly as mentioned above. If you are unable to find any information, please return "Not Found" for that key. DO NOT give any other commentry just return the JSON response. The JSON response should be valid and well formatted. I will provide you with a company name and you will search the web for all the information related to the company. The company name is:
    {company_name}
    """
    completion = client.chat.completions.create(
        model="gpt-4o-search-preview",
        web_search_options={
            "search_context_size": "high",   # only this key is supported
            "user_location": {                          # bias results toward India
                "type": "approximate",
                "approximate": {
                    "country": "IN"
                }
            },
        # "search_context_size": "medium",  # only this key is supported
        },
        messages=[
            {
                "role": "user",
                "content": prompt
            },
        ],
    )
    # print(completion.choices[0].message.content)
    response = completion.choices[0].message.content
    # Parse the JSON response
    try:
        report = parse_model_json(response)
    except ValueError as e:
        print(f"[Warning] Couldn't parse overview report JSON for '{company_name}': {e}")
        # If parsing fails, try to extract the first valid JSON object
        report = {
            "CIN Number": "Not Found",
            "Name": "Not Found",
            "Listed on Stock Exchange": "Not Found",
            "Company Status (for eFiling)": "Not Found",
            "PAN": "Not Found",
            "Date of Incorporation": "Not Found",
            "LEI Number": "Not Found",
            "LEI Expiry Date": "Not Found",
            "RoC Code": "Not Found",
            "email id": "Not Found",
            "Sector": "Not Found",
            "Industry": "Not Found",
            "website": "Not Found",
            "Address": "Not Found",
            "State": "Not Found",
            "Authorised Share Capital": "Not Found",
            "Paid-up Share Capital": "Not Found",
            "Date of Last Filed Balance Sheet": "Not Found",
            "Current Directors & Key Managerial Personnel": []
        }
    return report

def main():
    # Get the overview report for the company
    #company_name = "Riota Private Limited"
    company_name = "Value Point Systems Private Limited"
    response = get_overview_report(company_name)
    print("Response:")
    print(response)
if __name__ == "__main__":
    main()