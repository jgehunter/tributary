# Technology Stack

**Analysis Date:** 2026-01-19

## Languages

**Primary:**
- Python 3.11+ - All application code (`src/tributary/`)

**Secondary:**
- YAML - Configuration files (`config/*.yaml`)
- SQL - Database schema (`docker/questdb/init.sql`)
- Shell - Init scripts (`docker/questdb/init.sh`)

## Runtime

**Environment:**
- Python 3.11+ (specified in `pyproject.toml`: `requires-python = ">=3.11"`)

**Package Manager:**
- uv (indicated by `uv.lock` lockfile)
- Lockfile: present (`uv.lock`)

**Virtual Environment:**
- `.venv/` directory (local virtual environment)

## Frameworks

**Core:**
- Pydantic 2.0+ - Data validation and models (`src/tributary/core/models.py`)
- aiohttp 3.9+ - Async HTTP client for API requests (`src/tributary/collectors/polymarket/collector.py`)
- APScheduler 3.10+ - Task scheduling (`src/tributary/scheduler/scheduler.py`)

**CLI:**
- Click 8.0+ - Command-line interface framework (`src/tributary/cli/commands.py`)
- Rich 13.0+ - Terminal formatting and logging (`src/tributary/cli/commands.py`)

**Testing:**
- pytest 7.0+ - Test framework
- pytest-asyncio 0.23+ - Async test support
- pytest-cov 4.0+ - Coverage reporting

**Build/Dev:**
- setuptools 61.0+ - Package building (`pyproject.toml`)
- Black 24.0+ - Code formatting
- Ruff 0.1+ - Linting
- mypy 1.0+ - Type checking

## Key Dependencies

**Critical:**
- `py-clob-client>=0.17` - Official Polymarket CLOB API client (`src/tributary/collectors/polymarket/auth.py`)
- `eth-account>=0.10` - Ethereum wallet signing for Polymarket L2 auth
- `questdb>=2.0` - QuestDB ILP ingestion client (`src/tributary/storage/questdb.py`)
- `psycopg2-binary>=2.9` - PostgreSQL client for QuestDB queries

**Infrastructure:**
- `pyyaml>=6.0` - YAML configuration parsing (`src/tributary/core/config.py`)
- `python-dotenv>=1.0` - Environment variable loading (`src/tributary/core/config.py`)

**Optional (Notebooks):**
- `jupyter>=1.0` - Interactive notebooks
- `pandas>=2.0` - Data analysis
- `numpy>=1.26` - Numerical operations
- `matplotlib>=3.8`, `seaborn>=0.13`, `plotly>=5.0` - Visualization
- `sqlalchemy>=2.0` - Database ORM for analysis
- `scipy>=1.11` - Scientific computing

## Configuration

**Environment:**
- `.env` file loaded via `python-dotenv` at startup (`src/tributary/core/config.py`)
- Environment variable substitution in YAML: `${VAR:default}` syntax
- Key env vars:
  - `TRIBUTARY_ENV` - Environment selector (development/production)
  - `QUESTDB_HOST`, `QUESTDB_USER`, `QUESTDB_PASSWORD` - Database connection
  - `POLYMARKET_PRIVATE_KEY` - Wallet private key for API auth
  - `METRICS_PORT` - Monitoring endpoint port

**Build:**
- `pyproject.toml` - Package configuration, dependencies, tool settings
- `uv.lock` - Dependency lockfile
- `[tool.pytest.ini_options]` - pytest configuration in pyproject.toml
- `[tool.ruff]` and `[tool.black]` - Linting/formatting settings

**Application Config:**
- `config/base.yaml` - Base configuration (always loaded)
- `config/development.yaml` - Development overrides
- `config/markets/polymarket.yaml` - Market-specific configuration
- Config loading order: base.yaml -> {env}.yaml -> markets/*.yaml

## Platform Requirements

**Development:**
- Python 3.11+
- Docker/Docker Compose (for QuestDB)
- uv package manager (recommended)

**Production:**
- Docker Compose deployment
- QuestDB container (`questdb/questdb:latest`)
- Polygon wallet with private key for Polymarket auth
- Persistent volume for QuestDB data

**Infrastructure:**
- QuestDB ports: 9000 (HTTP/Console), 9009 (ILP TCP), 8812 (PostgreSQL)
- Metrics port: 8080 (configurable)

## Entry Points

**CLI:**
- `tributary` command defined in `pyproject.toml` -> `tributary.cli.commands:main`
- Alternative: `python -m tributary` via `src/tributary/__main__.py`

**Scripts:**
- `scripts/backfill.py` - Historical data backfill utility

---

*Stack analysis: 2026-01-19*
