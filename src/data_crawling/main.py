import asyncio
from typing import Any
from aws_lambda_powertools import Logger
from aws_lambda_powertools.utilities.typing import LambdaContext
from core.db.documents import GradeDocument
from crawlers import LoiGiaiHayCrawler
from dispatcher import CrawlerDispatcher

logger = Logger(service="mathpal/crawler")
_dispatcher = CrawlerDispatcher()
_dispatcher.register("loigiaihay", LoiGiaiHayCrawler)

async def handle(event, context: LambdaContext | None = None) -> dict[str, Any]:
    link = event.get("link", ""	)
    grade_name = event.get("grade_name", "")
    
    grade_id = GradeDocument.get_or_create(name=grade_name)
    crawler = _dispatcher.get_crawler(link)
    try:
        await crawler.extract(link=link, grade_id=grade_id)
        return {"statusCode": 200, "body": "Link processed successfully"}
    except Exception as e:
        return {"statusCode": 500, "body": f"An error occurred: {str(e)}"}

# AWS Lambda handler wrapper
def handler(event, context: LambdaContext | None = None) -> dict[str, Any]:
    return asyncio.run(handle(event, context))
    
    
if __name__ == "__main__":
    link = "https://loigiaihay.com/de-thi-vao-lop-6-mon-toan-truong-cau-giay-nam-2023-a142098.html"
    event = {
        "links": link,
        "grade_name": "grade_5"
    }
    asyncio.run(handle(event, None))