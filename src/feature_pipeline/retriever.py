import sys
from pathlib import Path
import os

# To mimic using multiple Python modules, such as 'core' and 'feature_pipeline',
# we will add the './src' directory to the PYTHONPATH. This is not intended for
# production use cases but for development and educational purposes.
ROOT_DIR = str(Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

# Change working directory to project root to ensure .env is loaded properly
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
os.chdir(PROJECT_ROOT)


from core import get_logger
from core.config import settings
from core.rag.retriever import VectorRetriever

logger = get_logger(__name__)

settings.patch_localhost()
logger.warning(
    "Patched settings to work with 'localhost' URLs. \
    Remove the 'settings.patch_localhost()' call from above when deploying or running inside Docker."
)

if __name__ == "__main__":
    query = """
Tôi đang học lớp 5
Tôi muốn tìm kiếm bài toán về biểu đồ hình
"""

    retriever = VectorRetriever(query=query)
    hits = retriever.retrieve_top_k(k=6, to_expand_to_n_queries=5)
    reranked_hits = retriever.rerank(hits=hits, keep_top_k=5)

    logger.info("====== RETRIEVED DOCUMENTS ======")
    for rank, hit in enumerate(reranked_hits):
        logger.info(f"Rank = {rank} : {hit}")
