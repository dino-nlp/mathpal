from typing import Optional, Tuple

from models.base import VectorDBDataModel
    
class ExamCleanedModel(VectorDBDataModel):
    entry_id: str
    type: str
    link: str
    grade_id: str
    cleaned_content: str
    
    def to_payload(self) -> Tuple[str, dict]:
        data = {
            "type": self.type,
            "link": self.link,
            "cleaned_content": self.cleaned_content,
            "grade_id": self.grade_id
        }
        return self.entry_id, data