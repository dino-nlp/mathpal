from typing import Tuple
import numpy as np
from models.base import VectorDBDataModel


class ExamEmbeddedChunkModel(VectorDBDataModel):
    entry_id: str
    type: str
    link: str
    grade_id: str
    chunk_id: str
    chunk_content: str
    embedded_content: np.ndarray
    
    class Config:
        arbitrary_types_allowed = True
        
    def to_payload(self) -> Tuple[str, np.ndarray, dict]:
        data = {
            "id": self.entry_id,
            "type": self.type,
            "link": self.link,
            "grade_id": self.grade_id,
            "content": self.chunk_content,
        }
        return self.chunk_id, self.embedded_content, data
    