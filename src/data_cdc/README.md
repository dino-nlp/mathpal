# Data CDC (Change Data Capture) Module

## Overview

The Data CDC module implements a real-time Change Data Capture system that monitors MongoDB collections for changes and publishes them to RabbitMQ for downstream processing. This enables event-driven data processing and ensures that new data is immediately available for the feature pipeline.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MongoDB       │    │   CDC Service   │    │   RabbitMQ      │
│   Collections   │───▶│   (Change       │───▶│   Message Queue │
│   (exam, etc.)  │    │    Streams)     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Key Components

### 1. CDC Service (`cdc.py`)
- **MongoDB Change Streams**: Monitors collections for real-time changes
- **Event Filtering**: Filters for specific operations (insert, update, delete)
- **Data Transformation**: Converts MongoDB documents to JSON format
- **RabbitMQ Publishing**: Sends events to message queue

### 2. Configuration (`config.py`)
- **MongoDB Connection**: Database connection settings
- **RabbitMQ Settings**: Message queue configuration
- **Queue Names**: Defines target queues for different data types

## 🚀 Features

- **Real-time Monitoring**: Instant detection of database changes
- **Event-driven Processing**: Triggers downstream pipelines automatically
- **Data Type Filtering**: Supports different collection types (exam, etc.)
- **Error Handling**: Robust error handling and logging
- **Scalable Architecture**: Can handle multiple collections and data types

## 📋 Supported Data Types

Currently supports the following MongoDB collections:
- **exam**: Math examination data with questions and solutions
- **Extensible**: Easy to add new collection types

## 🔧 Configuration

### Environment Variables
```bash
# MongoDB Configuration
MONGO_DATABASE_HOST=mongodb://mongo1:30001,mongo2:30002,mongo3:30003/?replicaSet=my-replica-set
MONGO_DATABASE_NAME=mathpal

# RabbitMQ Configuration
RABBITMQ_DEFAULT_USERNAME=guest
RABBITMQ_DEFAULT_PASSWORD=guest
RABBITMQ_HOST=mq
RABBITMQ_PORT=5673
RABBITMQ_QUEUE_NAME=mathpal_queue
```

## 🚀 Usage

### Running the CDC Service

#### Docker (Recommended)
```bash
# Start the CDC service via Docker Compose
make docker-start

# View CDC service logs
docker logs mathpal-data-cdc
```

#### Local Development
```bash
# Set up environment
export PYTHONPATH=$(pwd)/src

# Run CDC service
python src/data_cdc/cdc.py
```

### Event Flow

1. **Data Insertion**: New documents are inserted into MongoDB collections
2. **Change Detection**: CDC service detects changes via MongoDB Change Streams
3. **Event Processing**: Changes are filtered and transformed
4. **Message Publishing**: Events are published to RabbitMQ
5. **Downstream Processing**: Feature pipeline consumes messages for processing

## 📊 Event Structure

### MongoDB Change Event
```json
{
  "operationType": "insert",
  "ns": {
    "db": "mathpal",
    "coll": "exam"
  },
  "fullDocument": {
    "_id": "ObjectId(...)",
    "content": "Math problem content...",
    "grade_id": "grade_5",
    "metadata": {...}
  }
}
```

### Transformed Event
```json
{
  "type": "exam",
  "entry_id": "ObjectId_string",
  "content": "Math problem content...",
  "grade_id": "grade_5",
  "metadata": {...}
}
```

## 🔍 Monitoring and Logging

### Logging
The CDC service uses structured logging with the following log levels:
- **INFO**: Normal operation events
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors and exceptions

### Health Checks
- **MongoDB Connection**: Verifies database connectivity
- **RabbitMQ Connection**: Ensures message queue availability
- **Change Stream Status**: Monitors stream health

## 🛠️ Development

### Adding New Data Types

1. **Update Collection Filter**:
   ```python
   # In cdc.py, add new collection to supported types
   if data_type not in ["exam", "new_collection"]:
       logger.info(f"Unsupported data type: '{data_type}', skipping...")
       continue
   ```

2. **Add Configuration**:
   ```python
   # In config.py, add new queue configuration
   RABBITMQ_NEW_COLLECTION_QUEUE = "new_collection_queue"
   ```

3. **Update Event Processing**:
   ```python
   # Add specific processing logic for new data type
   if data_type == "new_collection":
       # Custom processing logic
       pass
   ```

### Testing

#### Unit Tests
```bash
# Run CDC unit tests
python -m pytest tests/data_cdc/ -v
```

#### Integration Tests
```bash
# Test with real MongoDB and RabbitMQ
docker-compose up -d mongo1 mongo2 mongo3 mq
python tests/integration/test_cdc_integration.py
```

## 🔧 Troubleshooting

### Common Issues

1. **MongoDB Connection Issues**
   - Verify MongoDB replica set is running
   - Check connection string format
   - Ensure network connectivity

2. **RabbitMQ Connection Issues**
   - Verify RabbitMQ service is running
   - Check credentials and permissions
   - Ensure queue exists

3. **Change Stream Errors**
   - Verify MongoDB version supports change streams
   - Check replica set configuration
   - Ensure proper permissions

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python src/data_cdc/cdc.py
```

## 📈 Performance Considerations

- **Memory Usage**: Change streams maintain cursor state
- **Network Latency**: Consider MongoDB and RabbitMQ proximity
- **Event Volume**: Monitor message queue depth
- **Error Recovery**: Implement retry mechanisms for failed events

## 🔗 Dependencies

- **MongoDB**: Document database with change streams
- **RabbitMQ**: Message queue system
- **Pika**: Python RabbitMQ client
- **Pymongo**: MongoDB Python driver
- **Structlog**: Structured logging

## 📚 Related Documentation

- [MongoDB Change Streams](https://docs.mongodb.com/manual/changeStreams/)
- [RabbitMQ Python Client](https://pika.readthedocs.io/)
- [PyMongo Documentation](https://pymongo.readthedocs.io/)

---

**Data CDC Module** - Real-time data ingestion for the MathPal platform 🔄📊
