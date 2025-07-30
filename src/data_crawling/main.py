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
    links = event.get("links", [])
    grade_name = event.get("grade_name", "")
    
    if len(links)==0 or not grade_name:
        return {"statusCode": 500, "body": f"An error occurred: len(links)==0 or grade_name is empty"}
    grade_id = GradeDocument.get_or_create(name=grade_name)
    crawler = _dispatcher.get_crawler(links[0])
    try:
        await crawler.extract(links=links, grade_id=grade_id)
        return {"statusCode": 200, "body": "Link processed successfully"}
    except Exception as e:
        return {"statusCode": 500, "body": f"An error occurred: {str(e)}"}

# AWS Lambda handler wrapper
def handler(event, context: LambdaContext | None = None) -> dict[str, Any]:
    return asyncio.run(handle(event, context))
    
    
if __name__ == "__main__":
    links = [
        "https://loigiaihay.com/de-thi-vao-lop-6-mon-toan-truong-cau-giay-nam-2023-a142098.html",
        "https://loigiaihay.com/de-thi-vao-lop-6-mon-toan-truong-luong-the-vinh-2021-co-dap-an-a134641.html",
        "https://loigiaihay.com/de-thi-vao-lop-6-mon-toan-truong-chuyen-ngoai-ngu-ha-noi-e30892.html",
        "https://loigiaihay.com/de-thi-vao-lop-6-mon-toan-truong-amsterdam-ha-noi-e30968.html",
        "https://loigiaihay.com/de-thi-vao-lop-6-mon-toan-truong-nguyen-tat-thanh-nam-2025-co-dap-an-a185630.html",
        "https://loigiaihay.com/de-thi-vao-lop-6-mon-toan-truong-archimedes-2022-bai-co-ban-co-dap-an-a155547.html",
    ]
    event = {
        "links": links,
        "grade_name": "grade_5"
    }
    asyncio.run(handle(event, None))