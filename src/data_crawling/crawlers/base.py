import time
from typing import List
from abc import ABC, abstractmethod
from tempfile import mkdtemp
from core.db.documents import BaseDocument
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher

class BaseCrawler(ABC):
    model: type[BaseDocument]

    @abstractmethod
    async def extract(self, links: List[str], **kwargs) -> None: ...


class BaseAbstractCrawler(BaseCrawler, ABC):
    def __init__(self) -> None:
        self.update_configure()
    
    def update_configure(self) -> None:
        pass 


    def login(self) -> None:
        pass