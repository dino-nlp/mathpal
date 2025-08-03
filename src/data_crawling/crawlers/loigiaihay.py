from typing import Optional
from pydantic import BaseModel, Field
from crawlers.base import BaseAbstractCrawler
from core.db.documents import ExamDocument
from aws_lambda_powertools import Logger
from core.config import settings
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, BrowserConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.async_dispatcher import SemaphoreDispatcher
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from crawl4ai import LLMExtractionStrategy
from crawl4ai import RateLimiter, LLMConfig
import traceback
import json

logger = Logger(service="mathpal/crawler/loigiaihay")

class Proplem(BaseModel):
    question: str = Field(..., description="Main question of the test, usually starting with 'Câu ...' or 'Bài...'. The question may contain LaTeX formulas")
    image_url: Optional[str] = Field(None, description="URL of the illustrative image, in case the question requires a diagram for description")
    solution: str = Field(..., description=" Solution guide, which may contain LaTeX formulas.")
    result: Optional[str] = Field(None, description="The answers to the multiple-choice questions can be A, B, C, D, or a final number.")

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

        llm_strategy = LLMExtractionStrategy(
            llm_config = LLMConfig(provider=f"openrouter/{settings.OPENROUTER_BASE_MODEL}", api_token=settings.OPENROUTER_KEY),
            schema=Proplem.model_json_schema(), # Or use model_json_schema()
            extraction_type="schema",
            instruction="""
            You are an expert Web Data Extraction agent. Your task is to parse the content of a provided exam and extract **ALL** the questions within it.
            
            **Instructions:**

            * Extract all questions sequentially, starting from the first question to the final question.
            * For each question, capture the following data points:
                * The full `question`.
                * Associated illustrative `image_url` (if any).
                * The `solution`.
                * The `result`.
            * All mathematical formulas within the questions and solutions **MUST** be formatted using LaTeX.
            * Ignore all non-question content, such as page introductions, comments, advertisements, or related links in the footer.
            * Format the final output to strictly follow the provided `JSON schema` structure.
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

    async def extract(self, link, **kwargs):
        logger.info(f"Starting scrapping: {link}")
        
        try:
            logger.info("Initializing AsyncWebCrawler...")
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                logger.info("AsyncWebCrawler initialized successfully")
                
                result = await crawler.arun(
                    url=link,
                    config=self.run_config,
                )
                
                if result.success:
                    # Kiểm tra extracted_content
                    if hasattr(result, 'extracted_content') and result.extracted_content:
                        logger.info(f"Found extracted content, processing...")
                        
                        # Parse extracted content thành list Proplem
                        try:
                            # Nếu extracted_content là string JSON
                            if isinstance(result.extracted_content, str):
                                extracted_data = json.loads(result.extracted_content)
                            else:
                                extracted_data = result.extracted_content
                            
                            # Xử lý data để tạo list Proplem
                            probloms = []
                            if isinstance(extracted_data, list):
                                # Nếu là list trực tiếp
                                for item in extracted_data:
                                    try:
                                        proplem = Proplem(**item)
                                        probloms.append(proplem)
                                    except Exception as e:
                                        logger.error(f"❌ Failed to create Proplem from item: {e}")
                            elif isinstance(extracted_data, dict):
                                # Nếu là dict, kiểm tra các keys phổ biến
                                for key in ['problems', 'questions', 'data', 'items']:
                                    if key in extracted_data and isinstance(extracted_data[key], list):
                                        for item in extracted_data[key]:
                                            try:
                                                proplem = Proplem(**item)
                                                probloms.append(proplem)
                                            except Exception as e:
                                                logger.error(f"❌ Failed to create Proplem from item: {e}")
                                        break
                                else:
                                    # Thử parse toàn bộ dict như 1 Proplem
                                    try:
                                        proplem = Proplem(**extracted_data)
                                        probloms.append(proplem)
                                    except Exception as e:
                                        logger.error(f"❌ Failed to create Proplem from dict: {e}")
                            
                            logger.info(f"🎯 Total Probloms extracted: {len(probloms)}")
                            
                            # Map Proplem thành ExamDocument và lưu vào MongoDB
                            saved_count = 0
                            for idx, proplem in enumerate(probloms):
                                try:
                                    # Tạo ExamDocument từ Proplem
                                    exam_doc = ExamDocument(
                                        question=proplem.question,
                                        image_url=proplem.image_url,
                                        solution=proplem.solution,
                                        result=proplem.result,
                                        grade_id=kwargs.get("grade_id")
                                    )
                                    
                                    # Lưu vào MongoDB
                                    exam_doc.save()
                                    saved_count += 1
                                    
                                except Exception as e:
                                    logger.error(f"❌ Failed to save Proplem {idx+1} to MongoDB: {e}")
                            
                            logger.info(f"💾 Successfully saved {saved_count}/{len(probloms)} Proplem objects to MongoDB")
                            
                        except Exception as e:
                            logger.error(f"❌ Error parsing extracted content: {e}")
                    else:
                        logger.warning("⚠️ No extracted_content found in result")
                else:
                    logger.error(f"Failed to crawl {result.url}: {result.error_message}")
                            
                logger.info("Crawling completed successfully")
                
        except Exception as e:
            logger.error(f"Error during crawling: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
            

            