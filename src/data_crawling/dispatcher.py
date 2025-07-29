import re
from core.logger_utils import get_logger
from crawlers.base import BaseCrawler

logger = get_logger(__file__)


class CrawlerDispatcher:
    def __init__(self) -> None:
        self._crawlers = {}

    def register(self, domain: str, crawler: type[BaseCrawler]) -> None:
        self._crawlers[r"https://(www\.)?{}.com/*".format(re.escape(domain))] = crawler

    def get_crawler(self, url: str) -> BaseCrawler:
        for pattern, crawler in self._crawlers.items():
            if re.match(pattern, url):
                return crawler()
        else:
            logger.warning(
                f"No crawler found for {url}"
            )

            raise NotImplementedError(f"Not implemented crawler for {url}")