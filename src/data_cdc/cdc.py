import json

from bson import json_util
from config import settings
from core.db.mongo import MongoDatabaseConnector
from core.logger_utils import get_logger
from core.mq import publish_to_rabbitmq

logger = get_logger(__file__)


def stream_process():
    try:
        logger.info("Starting CDC stream process...")
        client = MongoDatabaseConnector()
        db = client["mathpal"]
        logger.info("Connected to MongoDB successfully")

        logger.info("Starting to watch MongoDB change stream...")
        # Watch changes in a specific collection
        changes = db.watch([{"$match": {"operationType": {"$in": ["insert"]}}}])
        logger.info("MongoDB watch started, waiting for changes...")
        
        for change in changes:
            data_type = change["ns"]["coll"]
            entry_id = str(change["fullDocument"]["_id"])  # Convert ObjectId to string
            
            logger.info(f"Change detected in collection: {data_type}")

            change["fullDocument"].pop("_id")
            change["fullDocument"]["type"] = data_type
            change["fullDocument"]["entry_id"] = entry_id
            
            # logger.info(f"Processing change: {change}")

            if data_type not in ["exam", "grade"]:
                logger.info(f"Unsupported data type: '{data_type}', skipping...")
                continue

            # Use json_util to serialize the document
            data = json.dumps(change["fullDocument"], default=json_util.default)
            logger.info(
                f"Change detected and serialized for a data sample of type {data_type}."
            )

            # Send data to rabbitmq
            try:
                publish_to_rabbitmq(queue_name=settings.RABBITMQ_QUEUE_NAME, data=data)
                logger.info(f"Data of type '{data_type}' published to RabbitMQ successfully.")
            except Exception as mq_error:
                logger.error(f"Failed to publish to RabbitMQ: {mq_error}")

    except Exception as e:
        logger.error(f"An error occurred in CDC stream: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    logger.info("=== Starting MathPal CDC Service ===")
    stream_process()
