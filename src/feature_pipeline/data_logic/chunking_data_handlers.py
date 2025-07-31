import hashlib
from abc import ABC, abstractmethod
from models.base import DataModel
from models.chunk import ExamChunkModel
from models.clean import ExamCleanedModel
from utils.chunking import chunk_text


class ChunkingDataHandler(ABC):
    """
    Abstract class for all Chunking data handlers.
    All data transformations logic for the chunking step is done here
    """

    @abstractmethod
    def chunk(self, data_model: DataModel) -> list[DataModel]:
        pass
    
# class ExamChunkModel(DataModel):
#     entry_id: str
#     type: str
#     link: str
#     grade_id: str
#     chunk_id: str
#     chunk_content: str

class ExamChunkingHandler(ChunkingDataHandler):
    def chunk(self, data_model: ExamCleanedModel) -> list[ExamChunkModel]:
        data_model_list = []
        text_content = data_model.cleaned_content
        chunks = chunk_text(text_content)
        for chunk in chunks:
            model = ExamChunkModel(
                entry_id=data_model.entry_id,
                type=data_model.type,
                link=data_model.link,
                grade_id=data_model.grade_id,
                chunk_id=hashlib.md5(chunk.encode()).hexdigest(),
                chunk_content=chunk
            )
            data_model_list.append(model)
            
        return data_model_list