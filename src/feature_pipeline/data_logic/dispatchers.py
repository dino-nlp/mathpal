from core import get_logger
from models.base import DataModel
from models.raw import ExamRawModel
from pydantic import ValidationError

from data_logic.chunking_data_handlers import ExamChunkingHandler, ChunkingDataHandler
from data_logic.cleaning_data_handlers import ExamCleaningHandler, CleaningDataHandler
from data_logic.embedding_data_handlers import ExamEmbeddingHandler, EmbeddingDataHandler

logger = get_logger(__name__)

class RawDispatcher:
    @staticmethod
    def handle_mq_message(message: dict) -> DataModel:
        data_type = message.get("type")

        logger.info("Received message.", data_type=data_type)
        logger.debug("Message content.", message=message)

        if data_type == "exam":
            try:
                # Validate message structure before creating model
                # Required fields: DataModel (entry_id, type) + ExamRawModel (content, link, grade_id)
                required_fields = ['type', 'entry_id', 'question', 'solution', 'link', 'grade_id']
                missing_fields = [field for field in required_fields if field not in message]
                
                if missing_fields:
                    logger.error("Missing required fields in message.", missing_fields=missing_fields, message_keys=list(message.keys()))
                    raise ValueError(f"Missing required fields: {missing_fields}")
                
                message['content'] = f"### QUESTION: \n {message['question']} \n #### SOLUTION: \n {message['solution']}"
                return ExamRawModel(**message)
            except ValidationError as ve:
                logger.error("Validation error creating ExamRawModel.", 
                           error=str(ve), 
                           message=message,
                           error_details=ve.errors() if hasattr(ve, 'errors') else 'No error details available')
                # Re-raise with more context
                raise ValueError(f"Failed to validate ExamRawModel: {ve}")
            except Exception as e:
                logger.error("Unexpected error creating ExamRawModel.", 
                           error=str(e), 
                           error_type=type(e).__name__,
                           message=message)
                raise ValueError(f"Unexpected error creating ExamRawModel: {e}")
        else:
            logger.error("Unsupported data type.", data_type=data_type, message=message)
            raise ValueError(f"Unsupported data type: {data_type}")


class CleaningHandlerFactory:
    @staticmethod
    def create_handler(data_type) -> CleaningDataHandler:
        if data_type == "exam":
            return ExamCleaningHandler()
        else:
            raise ValueError("Unsupported data type")


class CleaningDispatcher:
    cleaning_factory = CleaningHandlerFactory()

    @classmethod
    def dispatch_cleaner(cls, data_model: DataModel) -> DataModel:
        data_type = data_model.type
        handler = cls.cleaning_factory.create_handler(data_type)
        clean_model = handler.clean(data_model)

        logger.info(
            "Data cleaned successfully.",
            data_type=data_type,
            cleaned_content_len=len(clean_model.cleaned_content),
        )

        return clean_model
    

class ChunkingHandlerFactory:
    @staticmethod
    def create_handler(data_type) -> ChunkingDataHandler:
        if data_type == "exam":
            return ExamChunkingHandler()
        else:
            raise ValueError("Unsupported data type")


class ChunkingDispatcher:
    chunking_factory = ChunkingHandlerFactory()

    @classmethod
    def dispatch_chunker(cls, data_model: DataModel) -> list[DataModel]:
        data_type = data_model.type
        handler = cls.chunking_factory.create_handler(data_type)
        chunk_models = handler.chunk(data_model)

        logger.info(
            "Cleaned content chunked successfully.",
            num=len(chunk_models),
            data_type=data_type,
        )

        return chunk_models
    

class EmbeddingHandlerFactory:
    @staticmethod
    def create_handler(data_type) -> EmbeddingDataHandler:
        if data_type == "exam":
            return ExamEmbeddingHandler()
        else:
            raise ValueError("Unsupported data type")


class EmbeddingDispatcher:
    embedding_factory = EmbeddingHandlerFactory()

    @classmethod
    def dispatch_embedder(cls, data_model: DataModel) -> DataModel:
        data_type = data_model.type
        handler = cls.embedding_factory.create_handler(data_type)
        embedded_chunk_model = handler.embedd(data_model)

        logger.info(
            "Chunk embedded successfully.",
            data_type=data_type,
            embedding_len=len(embedded_chunk_model.embedded_content),
        )

        return embedded_chunk_model
