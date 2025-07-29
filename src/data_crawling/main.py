import asyncio
from typing import Any
from core.config import settings
settings.patch_localhost()
from core.db.documents import GradeDocument
from crawlers import LoiGiaiHayCrawler
from dispatcher import CrawlerDispatcher
from utils import get_logger

logger = get_logger(__file__)
_dispatcher = CrawlerDispatcher()
_dispatcher.register("loigiaihay", LoiGiaiHayCrawler)

async def handle(event) -> dict[str, Any]:
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
    asyncio.run(handle(event=event))