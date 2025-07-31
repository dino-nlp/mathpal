from abc import ABC, abstractmethod

from models.base import DataModel
from models.chunk import ExamChunkModel
from models.embedded_chunk import ExamEmbeddedChunkModel
from utils.embedding import embedd_text


class EmbeddingDataHandler(ABC):
    """
    Abstract class for all embedding data handlers.
    All data transformations logic for the embedding step is done here
    """

    @abstractmethod
    def embedd(self, data_model: DataModel) -> DataModel:
        pass

class ExamEmbeddingHandler(EmbeddingDataHandler):
    def embedd(self, data_model: ExamChunkModel) -> ExamEmbeddedChunkModel:
        return ExamEmbeddedChunkModel(
            entry_id=data_model.entry_id,
            type=data_model.type,
            link=data_model.link,
            grade_id=data_model.grade_id,
            chunk_id=data_model.chunk_id,
            chunk_content=data_model.chunk_content,
            embedded_content=embedd_text(data_model.chunk_content)
        )