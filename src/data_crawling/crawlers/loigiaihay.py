from crawlers.base import BaseAbstractCrawler
from core.db.documents import ExamDocument
from core.logger_utils import get_logger
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, BrowserConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from crawl4ai import RateLimiter
import traceback

logger = get_logger(__file__)

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
        
        self.run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            css_selector="#sub-question-2",
            markdown_generator=DefaultMarkdownGenerator()
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
                        
                        if len(result.markdown) > 20:
                            instance = self.model(
                                content=result.markdown, link=result.url, grade_id=kwargs.get("grade_id")
                            )
                            instance.save()
                        else:
                            logger.info("CRAWLED CONTENT TOO SHORT")
                    else:
                        logger.error(f"Failed to crawl {result.url}: {result.error_message}")
                        
                logger.info("Crawling completed successfully")
                
        except Exception as e:
            logger.error(f"Error during crawling: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
            

            