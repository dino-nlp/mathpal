# MathPal - AI-Powered Math Education Platform

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-1.0+-blue.svg)](https://python-poetry.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

MathPal is an intelligent AI-powered math education platform designed to help Vietnamese students transition smoothly from 5th to 6th grade mathematics. The system combines advanced language models, RAG (Retrieval-Augmented Generation), and comprehensive data pipelines to provide personalized math tutoring and problem-solving assistance.

## 🏗️ System Architecture

MathPal is built as a microservices-based architecture with the following core components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Crawling │    │   Data CDC      │    │ Feature Pipeline│
│   (Web Scraping)│    │ (Change Stream) │    │ (Streaming)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MongoDB       │    │   RabbitMQ      │    │   Qdrant        │
│   (Raw Data)    │    │   (Message Q)   │    │   (Vector DB)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Training Pipeline│    │Inference Pipeline│
                       │ (Fine-tuning)   │    │ (RAG + LLM)     │
                       └─────────────────┘    └─────────────────┘
```

## 🚀 Key Features

- **Intelligent Data Crawling**: Automated web scraping of Vietnamese math education content
- **Real-time Data Processing**: CDC (Change Data Capture) for live data ingestion
- **Streaming Feature Pipeline**: Real-time data processing with Bytewax
- **Advanced RAG System**: Multi-query expansion, reranking, and semantic search
- **Fine-tuned LLM**: Custom-trained Gemma-3n model for Vietnamese math education
- **Comprehensive Evaluation**: Multi-metric performance assessment
- **Production-Ready**: Docker containerization and microservices architecture

## 📦 Project Structure

```
mathpal/
├── src/
│   ├── core/                    # Core utilities and configurations
│   ├── data_cdc/               # Change Data Capture service
│   ├── data_crawling/          # Web scraping and data collection
│   ├── feature_pipeline/       # Real-time data processing
│   ├── inference_pipeline/     # RAG and LLM inference
│   └── training_pipeline/      # Model fine-tuning
├── configs/                    # Configuration files
├── notebooks/                  # Jupyter notebooks for analysis
├── scripts/                    # Utility scripts
├── docker-compose.yml          # Infrastructure orchestration
└── Makefile                    # Development commands
```

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.11** - Primary programming language
- **Poetry** - Dependency management
- **Docker & Docker Compose** - Containerization
- **MongoDB** - Document database with replica set
- **RabbitMQ** - Message queue system
- **Qdrant** - Vector database for embeddings

### AI/ML Stack
- **Gemma-3n** - Base language model (Google)
- **Unsloth** - Efficient fine-tuning framework
- **Sentence Transformers** - Embedding models
- **Bytewax** - Stream processing framework
- **LangChain** - RAG framework
- **OPIK** - Performance tracking and optimization

### Development Tools
- **CometML** - Experiment tracking
- **Hugging Face** - Model hosting and datasets
- **AWS Lambda** - Serverless functions
- **Selenium** - Web scraping
- **Pydantic** - Data validation

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Poetry
- GPU (recommended for training and inference)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mathpal
   ```

2. **Install dependencies**
   ```bash
   make install
   ```

3. **Setup environment**
   ```bash
   cp env.example .env
   # Edit .env with your API keys and configurations
   make setup-env
   ```

4. **Start infrastructure**
   ```bash
   make docker-start
   ```

### Basic Usage

#### Data Pipeline
```bash
# Start data crawling
make crawl

# Process data through feature pipeline
make process
```

#### Training
```bash
# Quick training test
make train-quick

# Full production training
make train
```

#### Evaluation
```bash
# Quick evaluation
make evaluate-quick

# Full evaluation with progress tracking
make evaluate-llm-progress
```

## 📊 System Components

### 1. Data Crawling (`data_crawling/`)
Automated web scraping system for Vietnamese math education websites:
- **LoiGiaiHay Crawler**: Extracts math problems and solutions
- **AWS Lambda Integration**: Serverless deployment
- **Grade-based Organization**: Structured data by educational level

### 2. Change Data Capture (`data_cdc/`)
Real-time data ingestion from MongoDB:
- **MongoDB Change Streams**: Monitors database changes
- **RabbitMQ Integration**: Publishes changes to message queue
- **Event-driven Architecture**: Triggers downstream processing

### 3. Feature Pipeline (`feature_pipeline/`)
Real-time data processing with Bytewax:
- **Stream Processing**: Continuous data transformation
- **Multi-stage Pipeline**: Raw → Clean → Chunk → Embed
- **Vector Storage**: Stores embeddings in Qdrant

### 4. Inference Pipeline (`inference_pipeline/`)
RAG-based question answering system:
- **Multi-query Expansion**: Generates multiple search queries
- **Semantic Search**: Retrieves relevant documents
- **Reranking**: Optimizes result relevance
- **LLM Generation**: Produces final answers

### 5. Training Pipeline (`training_pipeline/`)
Model fine-tuning infrastructure:
- **Gemma-3n Fine-tuning**: Custom training for Vietnamese math
- **Experiment Tracking**: CometML integration
- **Configuration Management**: Flexible training configs
- **Evaluation Framework**: Comprehensive model assessment

## 🔧 Configuration

The system uses a hierarchical configuration system:

- **Environment Variables**: API keys and secrets
- **YAML Configs**: Training and pipeline parameters
- **Pydantic Settings**: Type-safe configuration validation

Key configuration files:
- `configs/production.yaml` - Production training settings
- `src/core/config.py` - Core application settings
- `.env` - Environment variables

## 📈 Performance & Monitoring

### Evaluation Metrics
- **Accuracy**: Question-answering correctness
- **Relevance**: Answer relevance to questions
- **Completeness**: Answer completeness scores
- **Progress Tracking**: Learning progression metrics

### Monitoring Tools
- **OPIK**: Performance tracking and optimization
- **CometML**: Experiment tracking and visualization
- **Structured Logging**: Comprehensive logging system

## 🐳 Deployment

### Docker Deployment
```bash
# Start all services
make docker-start

# View logs
make docker-logs

# Stop services
make docker-stop
```

### Production Considerations
- **GPU Requirements**: Minimum 16GB VRAM for training
- **Memory**: 32GB+ RAM recommended
- **Storage**: SSD storage for vector database
- **Network**: Stable internet for model downloads

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google** for the Gemma-3n model
- **Unsloth** for efficient fine-tuning
- **Vietnamese Math Education Community** for content and feedback

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Contact: ngohongthai.uet@gmail.com

---

**MathPal** - Empowering Vietnamese students with AI-powered math education 🧮✨
