from typing import Optional

from models.base import DataModel
    
class ExamRawModel(DataModel):
    content: dict
    link: str
    grade_id: str