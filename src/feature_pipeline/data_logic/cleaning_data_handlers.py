from abc import ABC, abstractmethod
from models.base import DataModel
from models.raw import ExamRawModel
from models.clean import ExamCleanedModel
from utils.cleaning import clean_text

class CleaningDataHandler(ABC):
    """
    Abstract class for all cleaning data handlers.
    All data transformations logic for the cleaning step is done here
    """

    @abstractmethod
    def clean(self, data_model: DataModel) -> DataModel:
        pass
    

class ExamCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: ExamRawModel) -> ExamCleanedModel:
        # Since content is now a string, use it directly instead of joining dict values
        content_text = data_model.content if data_model and data_model.content else ""
        
        return ExamCleanedModel(
            entry_id=data_model.entry_id,
            type=data_model.type,
            link=data_model.link,
            grade_id=data_model.grade_id,
            cleaned_content=clean_text(content_text)
        )