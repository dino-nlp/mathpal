from typing import Optional

from models.base import DataModel
    
class ExamRawModel(DataModel):
    content: str  # Changed from dict to str to match ExamDocument
    link: str
    grade_id: str