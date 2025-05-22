# Company Research Agent

A Python-based tool for extracting and analyzing legal cases related to companies from IndianKanoon.

## Features

- Automated search for legal cases on IndianKanoon
- Intelligent case relevance analysis using Gemini AI
- Structured data extraction using GPT-4
- Support for EC2 deployment
- Automated dependency checking and setup

## Prerequisites

- Python 3.8+
- Chrome/Chromium browser
- Required system libraries (for EC2 deployment)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/company-research-agent.git
cd company-research-agent
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. For EC2 deployment, install system dependencies:
```bash
sudo apt-get update
sudo apt-get install -y google-chrome-stable
sudo apt-get install -y libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2
```

## Configuration

1. Set up your API keys in `kanoon_ec2.py`:
```python
GOOGLE_API_KEY = "your_gemini_api_key_here"
OPENAI_API_KEY = "your_openai_api_key_here"
TAVILY_API_KEY = "your_tavily_api_key_here"
```

## Usage

Run the script with:
```bash
python kanoon_ec2.py
```

The script will:
1. Search for legal cases related to the specified company
2. Analyze case relevance
3. Extract structured data
4. Save results in JSON format

## Output

Results are saved in the `legal_content` directory:
- Raw case content: `legal_content/{company_name}_{case_id}.txt`
- Structured data: `legal_content/{company_name}_cases.json`

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 