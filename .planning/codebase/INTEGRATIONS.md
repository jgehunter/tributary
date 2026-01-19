# External Integrations

**Analysis Date:** 2026-01-19

## APIs & External Services

**Polymarket (Primary Data Source):**

Three APIs are used for data collection:

1. **CLOB API** (`https://clob.polymarket.com`)
   - Purpose: Real-time orderbook data
   - Endpoints: `/book` (orderbook by token_id)
   - SDK/Client: `py-clob-client` for authentication, `aiohttp` for requests
   - Auth: L2 API credentials derived from wallet signature
   - Rate limit: 1500 requests/10s (`clob_book`)
   - Implementation: `src/tributary/collectors/polymarket/collector.py`

2. **Gamma API** (`https://gamma-api.polymarket.com`)
   - Purpose: Market metadata and discovery
   - Endpoints: `/markets` (list, search, lookup by slug)
   - SDK/Client: `aiohttp` (no auth required for public endpoints)
   - Auth: None required
   - Rate limit: 300 requests/10s (`gamma_markets`)
   - Implementation: `src/tributary/collectors/polymarket/collector.py`

3. **Data API** (`https://data-api.polymarket.com`)
   - Purpose: Historical trades
   - Endpoints: `/trades` (by market condition_id)
   - SDK/Client: `aiohttp` (no auth required)
   - Auth: None required
   - Rate limit: 200 requests/10s (`data_trades`)
   - Implementation: `src/tributary/collectors/polymarket/collector.py`

**Polygon Network (Blockchain):**
- Purpose: Wallet authentication for Polymarket L2
- SDK/Client: `eth-account` for signing
- Auth: Polygon wallet private key (`POLYMARKET_PRIVATE_KEY`)
- Implementation: `src/tributary/collectors/polymarket/auth.py`

## Data Storage

**Databases:**

**QuestDB (Primary Time-Series Database):**
- Type: Time-series database optimized for high-ingest scenarios
- Connection:
  - ILP HTTP: `http://{host}:{http_port}` (port 9000)
  - PostgreSQL wire: `postgresql://{user}:{password}@{host}:{pg_port}/qdb` (port 8812)
- Client: `questdb.ingress.Sender` for ILP writes, `psycopg2` for queries
- Env vars: `QUESTDB_HOST`, `QUESTDB_USER`, `QUESTDB_PASSWORD`
- Implementation: `src/tributary/storage/questdb.py`

**Schema Tables:**
- `markets` - Market metadata (monthly partition)
- `trades` - Trade records (daily partition, dedup on timestamp+trade_id)
- `orderbook_snapshots` - Orderbook snapshots (daily partition)
- `collection_metrics` - Operational metrics (daily partition)

**Schema Definition:** `docker/questdb/init.sql`

**File Storage:**
- None (all data stored in QuestDB)

**Caching:**
- In-memory rate limiter state (`src/tributary/collectors/polymarket/rate_limiter.py`)
- In-memory last trade timestamp tracking (`src/tributary/core/base_collector.py`)

## Authentication & Identity

**Polymarket L2 Authentication:**
- Provider: Custom (Polymarket's own L2 auth system)
- Implementation: `src/tributary/collectors/polymarket/auth.py`
- Flow:
  1. Load Polygon wallet private key from `POLYMARKET_PRIVATE_KEY`
  2. Initialize `ClobClient` with private key (chain_id: 137 for Polygon mainnet)
  3. Derive API credentials via `client.derive_api_key()` (L1 wallet signature)
  4. Re-initialize client with L2 API credentials (api_key, api_secret, passphrase)
- Credentials: API key, secret, passphrase (derived or pre-configured)
- Optional env vars: `POLYMARKET_API_KEY`, `POLYMARKET_API_SECRET`, `POLYMARKET_PASSPHRASE`

**Public Endpoints:**
- Gamma API and Data API work without authentication
- CLOB API public endpoints (like orderbook) work without auth
- Auth enables trading operations (not used for data collection)

## Monitoring & Observability

**Error Tracking:**
- Logging via Python stdlib `logging` module
- Rich console output for CLI (`rich.logging.RichHandler`)
- No external error tracking service

**Logs:**
- Console output with configurable log level (`config/base.yaml`: `log_level`)
- Rich formatting in CLI mode
- Standard format in script mode

**Metrics:**
- In-memory metrics tracking (`src/tributary/monitoring/metrics.py`)
- Metrics written to QuestDB `collection_metrics` table
- Tracked: collection counts, error rates, latency
- Metrics endpoint planned: `METRICS_PORT` (8080, not fully implemented)

**Health Checks:**
- Periodic health check task in scheduler (configurable interval)
- Implementation: `src/tributary/scheduler/scheduler.py::_health_check`
- Currently basic logging only

## CI/CD & Deployment

**Hosting:**
- Local/self-hosted deployment
- Docker Compose for QuestDB (`docker/docker-compose.yml`)

**CI Pipeline:**
- Not detected (no CI config files found)

**Container Services:**
- `questdb` - QuestDB time-series database
- `questdb-init` - One-shot container for schema initialization

**Docker Configuration:**
- `docker/docker-compose.yml` - Service definitions
- `docker/questdb/init.sql` - Database schema
- `docker/questdb/init.sh` - Initialization script

## Environment Configuration

**Required env vars:**
- `QUESTDB_HOST` - QuestDB hostname (default: localhost)
- `QUESTDB_USER` - QuestDB user (default: admin)
- `QUESTDB_PASSWORD` - QuestDB password (default: quest)

**Required for authenticated Polymarket access:**
- `POLYMARKET_PRIVATE_KEY` - Polygon wallet private key (without 0x prefix)

**Optional env vars:**
- `TRIBUTARY_ENV` - Environment name (default: development)
- `METRICS_PORT` - Metrics HTTP port (default: 8080)
- `POLYMARKET_API_KEY`, `POLYMARKET_API_SECRET`, `POLYMARKET_PASSPHRASE` - Pre-generated credentials

**Secrets location:**
- `.env` file in project root (not committed, see `.env.example`)
- Environment variables

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Rate Limiting

**Implementation:** `src/tributary/collectors/polymarket/rate_limiter.py`

**Pattern:** Sliding window rate limiter with async lock

**Configured Limits (per 10-second window):**
| Endpoint | Requests | Config Key |
|----------|----------|------------|
| CLOB Book | 1500 | `clob_book` |
| Gamma Markets | 300 | `gamma_markets` |
| Data Trades | 200 | `data_trades` |

**Configuration:** `config/base.yaml` under `collectors.polymarket.rate_limits`

## Data Flow Summary

```
Polymarket APIs                    Application                     QuestDB
+---------------+                 +---------------+               +---------+
| Gamma API     |--metadata------>| Collector     |--ILP write--->| markets |
| (markets)     |                 |               |               |         |
+---------------+                 |               |               +---------+
                                  |               |
+---------------+                 |               |               +---------+
| Data API      |--trades-------->|               |--ILP write--->| trades  |
| (trades)      |                 |               |               |         |
+---------------+                 |               |               +---------+
                                  |               |
+---------------+                 |               |               +---------+
| CLOB API      |--orderbooks---->|               |--ILP write--->| order-  |
| (book)        |                 |               |               | book_   |
+---------------+                 +---------------+               | snap-   |
                                                                  | shots   |
                                                                  +---------+
```

---

*Integration audit: 2026-01-19*
