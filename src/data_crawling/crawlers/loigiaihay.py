from typing import Optional
from pydantic import BaseModel, Field
from crawlers.base import BaseAbstractCrawler
from core.db.documents import ExamDocument
from core.logger_utils import get_logger
from core.config import settings
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, BrowserConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from crawl4ai import LLMExtractionStrategy
from crawl4ai import RateLimiter, LLMConfig
import traceback

logger = get_logger(__file__)

class Proplem(BaseModel):
    question: str = Field(..., description="Câu hỏi chính của bài thi, thường bắt đầu bằng 'Câu ...'. Câu hỏi có thể chứa các công thức latex")
    image_url: Optional[str] = Field(None, description="URL hình ảnh minh họa, trong trường hợp câu hỏi cần hình vẽ để mô tả")
    solution: str = Field(..., description="Hướng dẫn giải, trong hướng dẫn có thể có các công thức latex")
    result: Optional[str] = Field(None, description="Đáp án của những câu hỏi trắc nghiệm")

class LoiGiaiHayCrawler(BaseAbstractCrawler):
    model = ExamDocument
    
    def update_configure(self):
        self.browser_config = BrowserConfig(
            headless=True,  # Đổi sang True để tránh vấn đề display
            verbose=True,   # Bật verbose để debug
            extra_args=[
                "--no-sandbox",
                "--disable-dev-shm-usage", 
                "--disable-gpu",
                "--disable-extensions",
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                "--disable-backgrounding-occluded-windows",
                "--disable-ipc-flooding-protection"
            ]
        )

        self.dispatcher = MemoryAdaptiveDispatcher(
                memory_threshold_percent=90.0,  # Pause if memory exceeds this
                check_interval=1.0,             # How often to check memory
                max_session_permit=10,          # Maximum concurrent tasks
                rate_limiter=RateLimiter(       # Optional rate limiting
                    base_delay=(1.0, 2.0),
                    max_delay=30.0,
                    max_retries=2
                )
            )
        
        llm_strategy = LLMExtractionStrategy(
            llm_config = LLMConfig(provider=f"openrouter/{settings.OPENROUTER_BASE_MODEL}", api_token=settings.OPENROUTER_KEY),
            schema=Proplem.model_json_schema(), # Or use model_json_schema()
            extraction_type="schema",
            instruction="""
            Bạn là một chuyên gia trích xuất dữ liệu web. Nhiệm vụ của bạn là đọc nội dung của một đề thi được cung cấp và trích xuất TOÀN BỘ các câu hỏi có trong đó.
            - Hãy trích xuất tất cả các câu hỏi, bắt đầu từ 'Câu 1' cho đến câu cuối cùng.
            - Với mỗi câu hỏi, lấy đầy đủ nội dung câu hỏi, hình ảnh minh họa (nếu có), lời giải chi tiết và đáp án cuối cùng.
            - Các công thức toán học trong câu hỏi và lời giải PHẢI được định dạng bằng LaTeX.
            - Bỏ qua tất cả các nội dung không phải là câu hỏi như: lời giới thiệu đầu trang, các bình luận, quảng cáo, hoặc các link liên quan ở cuối trang.
            - Định dạng kết quả đầu ra theo đúng cấu trúc JSON schema đã được cung cấp.
            """,
            chunk_token_threshold=1000,
            overlap_rate=0.0,
            apply_chunking=True,
            # input_format="fit_markdown",   # or "html", "fit_markdown"
        )
        
        self.run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            css_selector="#sub-question-2",
            extraction_strategy=llm_strategy,
        )

    async def extract(self, links, **kwargs):
        logger.info(f"Starting scrapping loigiaihay.com article: {len(links)} links")
        
        try:
            logger.info("Initializing AsyncWebCrawler...")
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                logger.info("AsyncWebCrawler initialized successfully")
                logger.info("Starting arun_many...")
                
                results = await crawler.arun_many(
                    urls=links,
                    config=self.run_config,
                    dispatcher=self.dispatcher
                )
                
                logger.info(f"Received {len(results)} results")
                
                for i, result in enumerate(results):
                    logger.info(f"Processing result {i+1}/{len(results)}")
                    if result.success:
                        dr = result.dispatch_result
                        logger.info(f"URL: {result.url}")
                        logger.info(f"Memory: {dr.memory_usage:.1f}MB")
                        logger.info(f"Duration: {dr.end_time - dr.start_time}")
                        # Save data to database
                        logger.info(f"Content length: {len(result.markdown) if result.markdown else 0}")
                        logger.info(f"RESULT: \n {result}")
                        
                        # if len(result.markdown) > 20:
                        #     instance = self.model(
                        #         content=result.markdown, link=result.url, grade_id=kwargs.get("grade_id")
                        #     )
                        #     instance.save()
                        # else:
                        #     logger.info("CRAWLED CONTENT TOO SHORT")
                    else:
                        logger.error(f"Failed to crawl {result.url}: {result.error_message}")
                        
                logger.info("Crawling completed successfully")
                
        except Exception as e:
            logger.error(f"Error during crawling: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
            

            