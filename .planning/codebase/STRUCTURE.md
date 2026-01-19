# Codebase Structure

**Analysis Date:** 2026-01-19

## Directory Layout

```
tributary/
├── config/                     # YAML configuration files
│   ├── base.yaml               # Base configuration (always loaded)
│   ├── development.yaml        # Development overrides
│   └── markets/
│       └── polymarket.yaml     # Market-specific configs
├── docker/                     # Docker infrastructure
│   ├── docker-compose.yml      # QuestDB container setup
│   └── questdb/
│       ├── init.sh             # DB initialization script
│       └── init.sql            # Schema DDL
├── scripts/                    # Standalone utility scripts
│   └── backfill.py             # Historical data backfill
├── src/tributary/              # Main Python package
│   ├── __init__.py             # Package version
│   ├── __main__.py             # python -m entry point
│   ├── cli/                    # Command-line interface
│   ├── collectors/             # Data collection layer
│   ├── core/                   # Domain models & config
│   ├── monitoring/             # Metrics & alerting
│   ├── scheduler/              # Job scheduling
│   ├── storage/                # Database persistence
│   └── validation/             # Data validation
├── tests/                      # Test suite
│   ├── conftest.py             # Shared fixtures
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
├── pyproject.toml              # Project config & dependencies
└── .env.example                # Environment variable template
```

## Directory Purposes

**`config/`:**
- Purpose: All application configuration in YAML format
- Contains: Base config, environment-specific overrides, market configurations
- Key files: `base.yaml` (defaults), `development.yaml` (dev overrides), `markets/polymarket.yaml` (market list)

**`docker/`:**
- Purpose: Container infrastructure for QuestDB time-series database
- Contains: Docker Compose file, initialization scripts and SQL schema
- Key files: `docker-compose.yml`, `questdb/init.sql` (table definitions)

**`scripts/`:**
- Purpose: Standalone utility scripts that can be run independently
- Contains: Backfill script for historical data loading
- Key files: `backfill.py`

**`src/tributary/`:**
- Purpose: Main Python package containing all application code
- Contains: All source modules organized by layer/concern
- Key files: `__init__.py` (version), `__main__.py` (module entry)

**`src/tributary/cli/`:**
- Purpose: Click-based command-line interface
- Contains: Command definitions, argument handling, rich console output
- Key files: `commands.py` (main CLI)

**`src/tributary/collectors/`:**
- Purpose: Exchange-specific data collection implementations
- Contains: Registry, abstract base class implementation for Polymarket
- Key files: `registry.py` (factory), `polymarket/collector.py` (Polymarket impl)

**`src/tributary/core/`:**
- Purpose: Shared domain models, configuration, and exceptions
- Contains: Pydantic models, dataclass configs, exception hierarchy
- Key files: `models.py` (Trade, OrderBookSnapshot, Market), `config.py`, `exceptions.py`, `base_collector.py`

**`src/tributary/monitoring/`:**
- Purpose: Collection metrics tracking and gap detection
- Contains: Stats aggregation, gap alerting
- Key files: `metrics.py`, `alerting.py`

**`src/tributary/scheduler/`:**
- Purpose: APScheduler-based periodic job management
- Contains: Main scheduler orchestrating collection tasks
- Key files: `scheduler.py` (TributaryScheduler)

**`src/tributary/storage/`:**
- Purpose: Database persistence layer
- Contains: QuestDB ILP writer
- Key files: `questdb.py` (QuestDBWriter)

**`src/tributary/validation/`:**
- Purpose: Data quality validation
- Contains: Trade and OrderBook validators
- Key files: `validators.py`

**`tests/`:**
- Purpose: pytest test suite
- Contains: Unit tests, integration tests (placeholder), shared fixtures
- Key files: `conftest.py` (fixtures), `unit/test_models.py`, `unit/test_validators.py`

## Key File Locations

**Entry Points:**
- `src/tributary/__main__.py`: Module execution entry
- `src/tributary/cli/commands.py`: CLI commands (`main()` function)
- `scripts/backfill.py`: Standalone backfill script

**Configuration:**
- `config/base.yaml`: Default configuration values
- `config/development.yaml`: Development environment overrides
- `config/markets/polymarket.yaml`: Market slugs and discovery settings
- `pyproject.toml`: Project metadata, dependencies, tool configs

**Core Logic:**
- `src/tributary/core/base_collector.py`: Abstract collector interface
- `src/tributary/collectors/polymarket/collector.py`: Polymarket implementation
- `src/tributary/scheduler/scheduler.py`: Main scheduler
- `src/tributary/storage/questdb.py`: Database writer

**Testing:**
- `tests/conftest.py`: Shared pytest fixtures
- `tests/unit/`: Unit test modules

**Infrastructure:**
- `docker/docker-compose.yml`: QuestDB service definition
- `docker/questdb/init.sql`: Database schema

## Naming Conventions

**Files:**
- Python modules: `snake_case.py` (e.g., `base_collector.py`, `rate_limiter.py`)
- Config files: `snake_case.yaml` (e.g., `base.yaml`, `polymarket.yaml`)
- Package markers: `__init__.py`

**Directories:**
- Python packages: `snake_case` (e.g., `collectors`, `polymarket`)
- Special directories: lowercase (e.g., `config`, `docker`, `scripts`, `tests`)

**Classes:**
- `PascalCase` (e.g., `MarketDataCollector`, `QuestDBWriter`, `TributaryScheduler`)

**Functions/Methods:**
- `snake_case` (e.g., `fetch_trades`, `write_orderbook_snapshots`)
- Private: `_leading_underscore` (e.g., `_collect_orderbooks`, `_initialize_markets`)

**Constants/Enums:**
- `UPPER_SNAKE_CASE` for module-level constants
- Enum values: `UPPER_SNAKE_CASE` (e.g., `AssetType.PREDICTION_MARKET`)

## Where to Add New Code

**New Collector (Exchange):**
- Create directory: `src/tributary/collectors/{exchange_name}/`
- Files needed:
  - `__init__.py`: Export collector class
  - `collector.py`: Subclass `MarketDataCollector`
  - `transformers.py`: API response to model transformations
  - `auth.py`: Exchange authentication (if needed)
  - `rate_limiter.py`: Exchange-specific rate limits (if different)
- Register in `src/tributary/collectors/registry.py`

**New Domain Model:**
- Add to `src/tributary/core/models.py`
- Export in `src/tributary/core/__init__.py`

**New CLI Command:**
- Add function decorated with `@main.command()` in `src/tributary/cli/commands.py`

**New Storage Backend:**
- Create `src/tributary/storage/{backend}.py`
- Implement same interface as `QuestDBWriter`
- Export in `src/tributary/storage/__init__.py`

**New Validator:**
- Add class to `src/tributary/validation/validators.py`
- Export in `src/tributary/validation/__init__.py`

**New Configuration Section:**
- Add dataclass in `src/tributary/core/config.py`
- Add field to `AppConfig`
- Update `_build_config()` to parse from YAML

**New Test:**
- Unit tests: `tests/unit/test_{module}.py`
- Integration tests: `tests/integration/test_{feature}.py`
- Use fixtures from `tests/conftest.py`

## Special Directories

**`.planning/`:**
- Purpose: GSD planning and codebase documentation
- Generated: By GSD commands
- Committed: Yes (documentation)

**`.venv/`:**
- Purpose: Python virtual environment
- Generated: By uv/pip
- Committed: No (in .gitignore)

**`__pycache__/`:**
- Purpose: Python bytecode cache
- Generated: By Python interpreter
- Committed: No (in .gitignore)

**`.pytest_cache/`:**
- Purpose: pytest cache for test discovery/state
- Generated: By pytest
- Committed: No (in .gitignore)

**`*.egg-info/`:**
- Purpose: Package metadata for editable installs
- Generated: By setuptools/pip
- Committed: No (in .gitignore)

---

*Structure analysis: 2026-01-19*
