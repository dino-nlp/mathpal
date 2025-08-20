# Data Crawling Module

## Overview

The Data Crawling module provides automated web scraping capabilities for Vietnamese math education websites. It extracts structured data including math problems, solutions, and educational content, then stores them in MongoDB for further processing. The module is designed to work with AWS Lambda for serverless deployment and supports multiple crawler implementations.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Sources   │    │   Crawler       │    │   MongoDB       │
│   (loigiaihay,  │───▶│   Dispatcher    │───▶│   Collections   │
│    etc.)        │    │                 │    │   (exam, etc.)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   AWS Lambda    │
                       │   (Serverless)  │
                       └─────────────────┘
```

## 🔧 Key Components

### 1. Crawler Dispatcher (`dispatcher.py`)
- **Crawler Registration**: Manages multiple crawler implementations
- **URL Routing**: Routes URLs to appropriate crawlers based on domain
- **Error Handling**: Centralized error handling for all crawlers

### 2. Base Crawler (`crawlers/base.py`)
- **Abstract Interface**: Defines common crawler interface
- **Shared Utilities**: Common web scraping utilities
- **Error Handling**: Standardized error handling patterns

### 3. LoiGiaiHay Crawler (`crawlers/loigiaihay.py`)
- **Specialized Scraping**: Extracts math problems and solutions from loigiaihay.com
- **Content Parsing**: Parses HTML to extract structured data
- **Grade Classification**: Organizes content by educational grade level

### 4. Main Handler (`main.py`)
- **AWS Lambda Integration**: Serverless function entry point
- **Event Processing**: Handles Lambda events and triggers crawling
- **Response Formatting**: Returns standardized responses

## 🚀 Features

- **Multi-Source Support**: Extensible architecture for multiple websites
- **Serverless Deployment**: AWS Lambda integration for scalability
- **Grade-based Organization**: Structured data by educational level
- **Robust Error Handling**: Comprehensive error handling and logging
- **Content Validation**: Ensures data quality and completeness
- **Rate Limiting**: Respects website rate limits and robots.txt

## 📋 Supported Sources

### Currently Supported
- **LoiGiaiHay**: Vietnamese math education website (loigiaihay.com)
  - Math problems and solutions
  - Grade 5-6 content
  - Exam preparation materials

### Extensible Architecture
The module is designed to easily add new crawlers for additional sources:
- **VnDoc**: Vietnamese document sharing platform
- **ToanMath**: Math education resources
- **Custom Sources**: Any Vietnamese math education website

## 🔧 Configuration

### Environment Variables
```bash
# MongoDB Configuration
MONGO_DATABASE_HOST=mongodb://mongo1:30001,mongo2:30002,mongo3:30003/?replicaSet=my-replica-set
MONGO_DATABASE_NAME=mathpal

# AWS Configuration (for Lambda deployment)
AWS_REGION=ap-southeast-2
AWS_ACCESS_KEY=your_access_key
AWS_SECRET_KEY=your_secret_key

# Crawler Configuration
CRAWLER_USER_AGENT=Mozilla/5.0 (compatible; MathPal/1.0)
CRAWLER_DELAY=1.0  # Delay between requests in seconds
CRAWLER_TIMEOUT=30  # Request timeout in seconds
```

### Crawler Settings
```python
# In crawler configuration
CRAWLER_SETTINGS = {
    "loigiaihay": {
        "base_url": "https://loigiaihay.com",
        "selectors": {
            "question": ".question-content",
            "solution": ".solution-content",
            "grade": ".grade-info"
        },
        "rate_limit": 1.0  # Requests per second
    }
}
```

## 🚀 Usage

### Running Crawlers

#### AWS Lambda Deployment
```bash
# Deploy to AWS Lambda
aws lambda create-function \
    --function-name mathpal-crawler \
    --runtime python3.11 \
    --handler main.handler \
    --zip-file fileb://crawler-package.zip

# Invoke function
aws lambda invoke \
    --function-name mathpal-crawler \
    --payload '{"link": "https://loigiaihay.com/example", "grade_name": "grade_5"}' \
    response.json
```

#### Local Development
```bash
# Set up environment
export PYTHONPATH=$(pwd)/src

# Run crawler locally
python src/data_crawling/main.py
```

#### Docker Deployment
```bash
# Start crawler service
make docker-start

# View crawler logs
docker logs mathpal-data-crawlers
```

### Crawler Invocation

#### Single URL Processing
```python
# Example Lambda event
event = {
    "link": "https://loigiaihay.com/de-thi-vao-lop-6-mon-toan-truong-cau-giay-nam-2023-a142098.html",
    "grade_name": "grade_5"
}

# Process the URL
result = handler(event, None)
```

#### Batch Processing
```python
# Process multiple URLs
urls = [
    "https://loigiaihay.com/url1",
    "https://loigiaihay.com/url2",
    "https://loigiaihay.com/url3"
]

for url in urls:
    event = {"link": url, "grade_name": "grade_5"}
    result = handler(event, None)
```

## 📊 Data Structure

### Extracted Content Format
```json
{
  "_id": "ObjectId(...)",
  "type": "exam",
  "grade_id": "grade_5",
  "content": "### QUESTION:\n[Math problem content]\n\n#### SOLUTION:\n[Solution content]",
  "metadata": {
    "source": "loigiaihay",
    "url": "https://loigiaihay.com/...",
    "title": "Math problem title",
    "difficulty": "medium",
    "topics": ["algebra", "geometry"],
    "crawled_at": "2024-01-01T00:00:00Z"
  },
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

## 🛠️ Development

### Adding New Crawlers

1. **Create Crawler Class**:
   ```python
   # src/data_crawling/crawlers/newsource.py
   from .base import BaseCrawler
   
   class NewSourceCrawler(BaseCrawler):
       def __init__(self):
           super().__init__("newsource")
       
       async def extract(self, link: str, grade_id: str):
           # Implement extraction logic
           pass
   ```

2. **Register Crawler**:
   ```python
   # In main.py
   from crawlers.newsource import NewSourceCrawler
   
   _dispatcher.register("newsource", NewSourceCrawler)
   ```

3. **Add URL Pattern**:
   ```python
   # In dispatcher.py
   def get_crawler(self, url: str):
       if "newsource.com" in url:
           return self._crawlers["newsource"]
   ```

### Testing

#### Unit Tests
```bash
# Run crawler unit tests
python -m pytest tests/data_crawling/ -v
```

#### Integration Tests
```bash
# Test with real websites
python tests/integration/test_crawler_integration.py
```

#### Mock Testing
```bash
# Test with mock responses
python tests/unit/test_crawler_mock.py
```

## 🔍 Monitoring and Logging

### Logging Levels
- **INFO**: Normal crawling operations
- **WARNING**: Rate limiting, retries
- **ERROR**: Failed requests, parsing errors
- **DEBUG**: Detailed request/response information

### Metrics
- **Success Rate**: Percentage of successful extractions
- **Response Time**: Average time per request
- **Error Rate**: Frequency of errors by type
- **Content Quality**: Validation of extracted content

## 🔧 Troubleshooting

### Common Issues

1. **Website Changes**
   - Update CSS selectors in crawler configuration
   - Verify website structure hasn't changed
   - Check for anti-bot measures

2. **Rate Limiting**
   - Increase delay between requests
   - Implement exponential backoff
   - Use proxy rotation if necessary

3. **Content Parsing Errors**
   - Validate HTML structure
   - Check for dynamic content loading
   - Update parsing logic for new formats

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export CRAWLER_DEBUG=true
python src/data_crawling/main.py
```

## 📈 Performance Considerations

- **Concurrency**: Limit concurrent requests to avoid overwhelming servers
- **Caching**: Implement caching for repeated requests
- **Respectful Crawling**: Follow robots.txt and rate limits
- **Error Recovery**: Implement retry mechanisms with exponential backoff

## 🔗 Dependencies

- **Selenium**: Web browser automation
- **BeautifulSoup**: HTML parsing
- **Requests**: HTTP client
- **Pymongo**: MongoDB driver
- **AWS Lambda Powertools**: Serverless utilities

## 📚 Related Documentation

- [Selenium Documentation](https://selenium-python.readthedocs.io/)
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/)
- [AWS Lambda Python](https://docs.aws.amazon.com/lambda/latest/dg/python-programming-model.html)
- [Web Scraping Best Practices](https://www.scrapehero.com/how-to-prevent-getting-blacklisted-while-scraping/)

## 🤝 Ethical Considerations

- **Respect robots.txt**: Always check and follow robots.txt files
- **Rate Limiting**: Implement appropriate delays between requests
- **Terms of Service**: Review and comply with website terms
- **Data Usage**: Use extracted data responsibly and ethically
- **Attribution**: Provide proper attribution to content sources

---

**Data Crawling Module** - Intelligent web scraping for Vietnamese math education 🕷️📚
