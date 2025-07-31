from typing import Optional

from models.base import DataModel

class ExamChunkModel(DataModel):
    entry_id: str
    type: str
    link: str
    grade_id: str
    chunk_id: str
    chunk_content: str