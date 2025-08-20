# Feature Pipeline Module

## Overview

The Feature Pipeline module implements a real-time data processing system using Bytewax for stream processing. It transforms raw data from RabbitMQ into structured, cleaned, chunked, and embedded content that is stored in Qdrant vector database. The pipeline supports multiple data types and provides a scalable foundation for the RAG (Retrieval-Augmented Generation) system.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RabbitMQ      │    │   Bytewax       │    │   Qdrant        │
│   (Raw Data)    │───▶│   Stream        │───▶│   (Vector DB)   │
│                 │    │   Processing    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Multi-stage   │
                       │   Pipeline      │
                       │   (Raw→Clean→   │
                       │    Chunk→Embed) │
                       └─────────────────┘
```

## 🔧 Key Components

### 1. Main Pipeline (`main.py`)
- **Bytewax Dataflow**: Defines the streaming processing pipeline
- **Operator Chain**: Orchestrates data transformation stages
- **Output Management**: Handles data storage to Qdrant

### 2. Data Flow (`data_flow/`)
- **Stream Input** (`stream_input.py`): RabbitMQ message consumption
- **Stream Output** (`stream_output.py`): Qdrant data insertion
- **Message Handling**: JSON message parsing and validation

### 3. Data Logic (`data_logic/`)
- **Dispatchers**: Route data to appropriate processors
  - `RawDispatcher`: Initial message processing
  - `CleaningDispatcher`: Content cleaning and normalization
  - `ChunkingDispatcher`: Text chunking for RAG
  - `EmbeddingDispatcher`: Vector embedding generation

### 4. Data Models (`models/`)
- **Base Models**: Abstract base classes for data structures
- **Raw Data**: Unprocessed input data models
- **Cleaned Data**: Normalized and cleaned content
- **Chunked Data**: Text chunks for retrieval
- **Embedded Data**: Vector embeddings with metadata

### 5. Utilities (`utils/`)
- **Chunking Utilities**: Text segmentation algorithms
- **Cleaning Utilities**: Content normalization and cleaning
- **Embedding Utilities**: Vector generation and management

## 🚀 Features

- **Real-time Processing**: Continuous stream processing with Bytewax
- **Multi-stage Pipeline**: Raw → Clean → Chunk → Embed transformation
- **Scalable Architecture**: Horizontal scaling with Bytewax
- **Fault Tolerance**: Error handling and recovery mechanisms
- **Flexible Data Types**: Support for multiple content types
- **Vector Storage**: Efficient storage in Qdrant vector database
- **Monitoring**: Comprehensive logging and metrics

## 📋 Pipeline Stages

### 1. Raw Data Processing
- **Message Consumption**: Read from RabbitMQ queues
- **JSON Parsing**: Parse incoming message format
- **Validation**: Ensure data integrity and structure
- **Type Detection**: Identify content type for routing

### 2. Content Cleaning
- **Text Normalization**: Standardize text formatting
- **HTML Cleaning**: Remove HTML tags and formatting
- **Noise Removal**: Filter out irrelevant content
- **Quality Validation**: Ensure content meets quality standards

### 3. Text Chunking
- **Semantic Chunking**: Split content by semantic boundaries
- **Size Optimization**: Create optimal chunk sizes for retrieval
- **Overlap Management**: Maintain context between chunks
- **Metadata Preservation**: Preserve chunk-level metadata

### 4. Vector Embedding
- **Model Loading**: Load embedding models (BAAI/bge-m3)
- **Batch Processing**: Efficient batch embedding generation
- **Dimension Management**: Handle embedding dimensions
- **Storage Optimization**: Optimize vector storage format

## 🔧 Configuration

### Environment Variables
```bash
# RabbitMQ Configuration
RABBITMQ_DEFAULT_USERNAME=guest
RABBITMQ_DEFAULT_PASSWORD=guest
RABBITMQ_HOST=mq
RABBITMQ_PORT=5673

# Qdrant Configuration
QDRANT_DATABASE_HOST=qdrant
QDRANT_DATABASE_PORT=6333
QDRANT_CLOUD_URL=your_qdrant_cloud_url
USE_QDRANT_CLOUD=false
QDRANT_APIKEY=your_api_key

# Embedding Configuration
EMBEDDING_MODEL_ID=BAAI/bge-m3
EMBEDDING_MODEL_MAX_INPUT_LENGTH=512
EMBEDDING_SIZE=1024
EMBEDDING_MODEL_DEVICE=cpu

# Bytewax Configuration
BYTEWAX_PYTHON_FILE_PATH=main:flow
DEBUG=false
BYTEWAX_KEEP_CONTAINER_ALIVE=true
```

### Pipeline Settings
```python
# Pipeline configuration
PIPELINE_CONFIG = {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "embedding_batch_size": 32,
    "max_retries": 3,
    "timeout": 30
}
```

## 🚀 Usage

### Running the Pipeline

#### Docker Deployment (Recommended)
```bash
# Start the feature pipeline
make docker-start

# View pipeline logs
docker logs mathpal-feature-pipeline
```

#### Local Development
```bash
# Set up environment
export PYTHONPATH=$(pwd)/src
export BYTEWAX_PYTHON_FILE_PATH=main:flow

# Run pipeline locally
python -m bytewax.run src/feature_pipeline/main.py
```

#### Bytewax CLI
```bash
# Run with Bytewax CLI
bytewax.run src/feature_pipeline/main.py
```

### Pipeline Monitoring

#### Logs
```bash
# View real-time logs
docker logs -f mathpal-feature-pipeline

# Filter by log level
docker logs mathpal-feature-pipeline | grep "ERROR"
```

#### Metrics
- **Processing Rate**: Messages processed per second
- **Error Rate**: Failed processing attempts
- **Latency**: End-to-end processing time
- **Queue Depth**: RabbitMQ queue monitoring

## 📊 Data Flow

### Input Message Format
```json
{
  "type": "exam",
  "entry_id": "507f1f77bcf86cd799439011",
  "content": "### QUESTION:\n[Math problem]\n\n#### SOLUTION:\n[Solution]",
  "grade_id": "grade_5",
  "metadata": {
    "source": "loigiaihay",
    "url": "https://example.com",
    "crawled_at": "2024-01-01T00:00:00Z"
  }
}
```

### Processing Stages

1. **Raw Processing**:
   ```python
   # Extract and validate message
   raw_data = RawData(
       content=message["content"],
       metadata=message["metadata"],
       type=message["type"]
   )
   ```

2. **Cleaning**:
   ```python
   # Clean and normalize content
   cleaned_data = CleanedData(
       content=clean_text(raw_data.content),
       metadata=raw_data.metadata,
       quality_score=calculate_quality(raw_data.content)
   )
   ```

3. **Chunking**:
   ```python
   # Split into semantic chunks
   chunks = [
       ChunkedData(
           content=chunk_text,
           metadata=cleaned_data.metadata,
           chunk_id=f"{entry_id}_{i}"
       )
       for i, chunk_text in enumerate(semantic_chunks)
   ]
   ```

4. **Embedding**:
   ```python
   # Generate vector embeddings
   embedded_data = EmbeddedData(
       content=chunk.content,
       embedding=generate_embedding(chunk.content),
       metadata=chunk.metadata,
       vector_id=chunk.chunk_id
   )
   ```

### Output Storage
```python
# Store in Qdrant collections
# Clean data collection
qdrant.insert("clean_exams", cleaned_data)

# Vector collection
qdrant.insert("vector_exams", embedded_data)
```

## 🛠️ Development

### Adding New Data Types

1. **Create Data Models**:
   ```python
   # src/feature_pipeline/models/new_type.py
   from .base import BaseModel
   
   class NewTypeData(BaseModel):
       content: str
       metadata: dict
       new_field: str
   ```

2. **Add Dispatcher Logic**:
   ```python
   # In data_logic/dispatchers.py
   class NewTypeDispatcher:
       @staticmethod
       def dispatch_processor(data: dict):
           if data["type"] == "new_type":
               return process_new_type(data)
   ```

3. **Update Pipeline**:
   ```python
   # In main.py, add new processing stage
   stream = op.map("new_type_dispatch", stream, NewTypeDispatcher.dispatch_processor)
   ```

### Custom Processors

#### Text Cleaning
```python
# src/feature_pipeline/utils/cleaning.py
def custom_cleaner(text: str) -> str:
    """Custom text cleaning logic"""
    # Remove specific patterns
    text = re.sub(r'pattern', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text
```

#### Chunking Strategy
```python
# src/feature_pipeline/utils/chunking.py
def semantic_chunker(text: str, chunk_size: int = 512) -> List[str]:
    """Semantic text chunking"""
    # Implement semantic chunking logic
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + sentence) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
```

### Testing

#### Unit Tests
```bash
# Run pipeline unit tests
python -m pytest tests/feature_pipeline/ -v
```

#### Integration Tests
```bash
# Test with real RabbitMQ and Qdrant
docker-compose up -d mq qdrant
python tests/integration/test_pipeline_integration.py
```

#### Performance Tests
```bash
# Load testing
python tests/performance/test_pipeline_load.py
```

## 🔍 Monitoring and Debugging

### Health Checks
- **RabbitMQ Connection**: Verify message queue connectivity
- **Qdrant Connection**: Check vector database availability
- **Embedding Model**: Validate model loading and inference
- **Pipeline Status**: Monitor processing stages

### Debug Mode
```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with debug output
python -m bytewax.run src/feature_pipeline/main.py --debug
```

### Performance Monitoring
```bash
# Monitor resource usage
docker stats mathpal-feature-pipeline

# Check queue depths
rabbitmqctl list_queues

# Monitor Qdrant metrics
curl http://localhost:6333/metrics
```

## 🔧 Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce batch sizes
   - Implement streaming processing
   - Monitor memory usage

2. **Processing Delays**
   - Check RabbitMQ queue depth
   - Verify Qdrant performance
   - Monitor embedding model speed

3. **Data Quality Issues**
   - Validate input data format
   - Check cleaning logic
   - Monitor quality scores

### Error Recovery
```python
# Implement retry logic
@retry(max_attempts=3, backoff_factor=2)
def process_with_retry(data):
    try:
        return process_data(data)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise
```

## 📈 Performance Optimization

### Scaling Strategies
- **Horizontal Scaling**: Add more Bytewax workers
- **Batch Processing**: Optimize batch sizes for embeddings
- **Caching**: Cache embedding model and frequent operations
- **Parallel Processing**: Use concurrent processing where possible

### Resource Management
- **Memory**: Monitor and optimize memory usage
- **CPU**: Efficient processing algorithms
- **GPU**: GPU acceleration for embedding generation
- **Network**: Optimize data transfer between services

## 🔗 Dependencies

- **Bytewax**: Stream processing framework
- **Qdrant**: Vector database
- **Sentence Transformers**: Embedding models
- **Pika**: RabbitMQ client
- **Pydantic**: Data validation
- **Structlog**: Structured logging

## 📚 Related Documentation

- [Bytewax Documentation](https://bytewax.io/docs)
- [Qdrant Python Client](https://qdrant.tech/documentation/guides/python/)
- [Sentence Transformers](https://www.sbert.net/)
- [RabbitMQ Python Client](https://pika.readthedocs.io/)

---

**Feature Pipeline Module** - Real-time data processing for intelligent math education ⚡🔄
