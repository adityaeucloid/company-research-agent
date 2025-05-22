# Company Research Agent UI

A Streamlit-based web interface for the Company Research Agent that allows you to extract company overview and financial data.

## Features

- Clean and intuitive user interface
- Company overview extraction
- Financial data extraction
- JSON download functionality
- Configurable search parameters

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure you're in the `ui` directory
2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. The application will be available at `http://localhost:8501`

## Usage

1. Enter the company name in the sidebar
2. Adjust the maximum results slider if needed
3. Choose between "Company Overview" or "Financial Data" tabs
4. Click the respective "Extract" button
5. View the results and download the JSON file if needed

## Configuration

The application uses the following environment variables:
- `GOOGLE_API_KEY`: Your Google API key for Gemini
- `OPENAI_API_KEY`: Your OpenAI API key
- `TAVILY_API_KEY`: Your Tavily API key

Make sure these are set in your environment or in a `.env` file.

## Running on EC2

1. Install dependencies:
```bash
sudo yum update -y
sudo yum install -y python3 python3-pip
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

3. Access the application through your EC2 instance's public IP address on port 8501

## Security Notes

- Make sure to configure your EC2 security group to allow inbound traffic on port 8501
- Consider using HTTPS in production
- Keep your API keys secure and never commit them to version control 