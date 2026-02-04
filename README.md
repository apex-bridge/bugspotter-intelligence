# BugSpotter Intelligence

RAG (Retrieval-Augmented Generation) service for intelligent bug analysis and deduplication.

[![Tests](https://github.com/apexbridge-tech/bugspotter-intelligence/workflows/Tests/badge.svg)](https://github.com/apexbridge-tech/bugspotter-intelligence/actions) [![Python](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org/) [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Features

- **Multi-LLM Support**: Extensible provider system supporting Ollama (local), Claude, and OpenAI
- **API Key Authentication**: Secure API key management with admin endpoints
- **Multi-Tenancy**: Tenant isolation for SaaS deployments
- **Rate Limiting**: Redis-based sliding window rate limiting
- **Semantic Deduplication**: Uses pgvector for finding similar bugs
- **RAG-Ready**: Context-aware prompt building for better AI responses
- **Async FastAPI**: High-performance async API
- **Docker-First**: PostgreSQL + pgvector + Ollama + Redis included
- **Full Test Coverage**: Unit and integration tests with testcontainers

## Prerequisites

- Python 3.12+
- Docker Desktop
- 8GB+ RAM (for local LLM)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/apexbridge-tech/bugspotter-intelligence.git
cd bugspotter-intelligence
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -e ".[dev]"
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env with your settings
```

### 5. Start Services
```bash
docker-compose up -d
```

Wait for Ollama to pull the model (first time ~5 minutes):
```bash
docker logs -f bugspotter-ollama
```

## Authentication Setup

### 1. Disable Auth for Initial Setup (Development)

Set in `.env`:
```env
AUTH_ENABLED=false
```

### 2. Create Admin API Key (Production)

With auth disabled, create your first admin key:

```bash
# Start the API
uvicorn bugspotter_intelligence.main:app --reload

# Create admin key via API
curl -X POST http://localhost:8000/api/v1/admin/api-keys \
  -H "Content-Type: application/json" \
  -d '{"name": "Initial Admin Key", "is_admin": true}'
```

Save the returned `plain_key` - it won't be shown again!

### 3. Enable Auth

Set in `.env`:
```env
AUTH_ENABLED=true
```

### 4. Use API Key

Include in all requests:
```bash
curl http://localhost:8000/api/v1/bugs/bug-123 \
  -H "Authorization: Bearer bsi_your_api_key_here"
```

## API Endpoints

### Bug Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/bugs/analyze` | Analyze and store a bug |
| GET | `/api/v1/bugs/{id}` | Get bug details |
| GET | `/api/v1/bugs/{id}/similar` | Find similar bugs |
| GET | `/api/v1/bugs/{id}/mitigation` | Get AI mitigation suggestion |
| PATCH | `/api/v1/bugs/{id}/resolution` | Update bug resolution |

### Admin (Requires Admin Key)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/admin/api-keys` | Create API key |
| GET | `/api/v1/admin/api-keys` | List API keys |
| GET | `/api/v1/admin/api-keys/{id}` | Get API key details |
| DELETE | `/api/v1/admin/api-keys/{id}` | Revoke API key |

### Other

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check (no auth) |
| POST | `/api/v1/ask` | Ask AI a question |

## Run Tests
```bash
# Unit tests only (fast)
pytest tests/ -v -m "not integration"

# All tests including integration (requires Docker)
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/bugspotter_intelligence --cov-report=term-missing
```

## Architecture
```
bugspotter-intelligence/
├── src/bugspotter_intelligence/
│   ├── api/                  # FastAPI routes and dependencies
│   │   ├── routes/           # Route handlers
│   │   └── deps.py           # Dependency injection
│   ├── auth/                 # Authentication module
│   │   ├── models.py         # APIKey, TenantContext
│   │   ├── repository.py     # Database access
│   │   ├── service.py        # Business logic
│   │   └── dependencies.py   # FastAPI dependencies
│   ├── db/                   # Database layer
│   │   ├── database.py       # Connection pool
│   │   ├── migrations.py     # Schema setup
│   │   └── bug_repository.py # Bug data access
│   ├── llm/                  # LLM provider abstraction
│   ├── rate_limiting/        # Redis rate limiting
│   ├── services/             # Business logic (CQRS)
│   └── config.py             # Pydantic settings
├── tests/                    # Test suite
├── docker/                   # Docker init scripts
└── docker-compose.yml        # Infrastructure
```

## Configuration

Key environment variables in `.env`:

```env
# Database
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=bugspotter_intelligence
DATABASE_USER=postgres
DATABASE_PASSWORD=postgres

# Authentication
AUTH_ENABLED=true
API_KEY_PREFIX=bsi_

# Redis (Rate Limiting)
REDIS_HOST=localhost
REDIS_PORT=6379

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_DEFAULT_RPM=60

# LLM Provider (ollama, claude, openai)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# Similarity Thresholds
SIMILARITY_THRESHOLD=0.75
DUPLICATE_THRESHOLD=0.90
```

See `.env.example` for all options.

## Docker Services

- **PostgreSQL 16** with pgvector extension
- **Ollama** with llama3.1:8b model (auto-downloaded)
- **Redis 7** for rate limiting

## Development Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed roadmap.

### Completed (v0.2.0)
- [x] LLM provider abstraction with registry pattern
- [x] Ollama, Claude, and OpenAI providers
- [x] Docker Compose setup with pgvector
- [x] FastAPI REST API routes
- [x] Bug similarity search
- [x] AI mitigation suggestions
- [x] API key authentication
- [x] Multi-tenant data isolation
- [x] Redis rate limiting
- [x] Comprehensive test suite

### Planned (v0.3.0+)
- [ ] Smart search with LLM reranking
- [ ] Query caching
- [ ] Feedback loop for improving suggestions
- [ ] Root cause analysis
- [ ] Trend detection

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest tests/ -v`)
5. Commit your changes (`git commit -m 'feat: add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details

## Acknowledgments

- [Ollama](https://ollama.ai/) - Local LLM runtime
- [Anthropic](https://anthropic.com/) - Claude API
- [OpenAI](https://openai.com/) - GPT API
- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity for PostgreSQL

## Contact

Apex Bridge Technology - [info@bugspotter.io](mailto:info@bugspotter.io)
