# Testing Patterns

**Analysis Date:** 2026-01-19

## Test Framework

**Runner:**
- pytest >= 7.0
- pytest-asyncio >= 0.23 (for async test support)
- pytest-cov >= 4.0 (for coverage)
- Config: `pyproject.toml`

**Assertion Library:**
- pytest native assertions
- Use `pytest.approx()` for floating point comparisons

**Configuration (from `pyproject.toml`):**
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Run Commands:**
```bash
pytest                      # Run all tests
pytest tests/unit           # Run unit tests only
pytest tests/integration    # Run integration tests only
pytest -v                   # Verbose output
pytest --cov=tributary      # With coverage
pytest -k "test_trade"      # Run tests matching pattern
```

## Test File Organization

**Location:**
- Separate `tests/` directory (not co-located with source)
- Mirrors source structure where applicable

**Naming:**
- Test files: `test_<module>.py`
- Test classes: `Test<ClassName>` (e.g., `TestTrade`, `TestTradeValidator`)
- Test methods: `test_<description>` (e.g., `test_trade_creation`, `test_price_too_low`)

**Structure:**
```
tests/
├── __init__.py              # Package marker
├── conftest.py              # Shared fixtures
├── unit/
│   ├── __init__.py
│   ├── test_models.py       # Tests for core/models.py
│   └── test_validators.py   # Tests for validation/validators.py
└── integration/
    └── __init__.py          # (no integration tests yet)
```

## Test Structure

**Suite Organization:**
```python
"""Tests for core models."""

from datetime import datetime

import pytest

from tributary.core.models import Trade, OrderBookSnapshot, Market, AssetType, Exchange, Side


class TestTrade:
    """Tests for Trade model."""

    def test_trade_creation(self):
        """Test basic trade creation."""
        trade = Trade(
            market_id="market-123",
            token_id="token-123",
            trade_id="trade-001",
            timestamp=datetime(2024, 1, 15, 12, 0, 0),
            price=0.75,
            size=100.0,
            side=Side.BUY,
        )
        assert trade.market_id == "market-123"
        assert trade.price == 0.75
        assert trade.side == Side.BUY

    def test_trade_value_computed(self):
        """Test that value is computed correctly."""
        trade = Trade(
            market_id="market-123",
            token_id="token-123",
            trade_id="trade-001",
            timestamp=datetime.utcnow(),
            price=0.50,
            size=200.0,
            side=Side.SELL,
        )
        assert trade.value == 100.0  # 0.50 * 200
```

**Patterns:**
- Group related tests in classes by component/model
- Each test class has a docstring describing what it tests
- Each test method has a docstring describing the specific scenario
- Use descriptive test names that explain the behavior being tested

## Fixtures

**Location:** `tests/conftest.py`

**Shared Fixtures:**
```python
@pytest.fixture
def sample_trade():
    """Create a sample trade for testing."""
    return Trade(
        market_id="0xabc123condition",  # conditionId (market-level)
        token_id="12345678901234567890",  # asset_id (outcome-level)
        trade_id="trade-001",
        timestamp=datetime.utcnow(),
        price=0.65,
        size=100.0,
        side=Side.BUY,
        is_buyer_maker=False,
    )


@pytest.fixture
def sample_orderbook():
    """Create a sample orderbook snapshot for testing."""
    return OrderBookSnapshot(
        market_id="0xabc123condition",
        token_id="12345678901234567890",
        timestamp=datetime.utcnow(),
        best_bid=0.64,
        best_ask=0.66,
        bid_size=500.0,
        ask_size=300.0,
        bid_prices=[0.64, 0.63, 0.62],
        bid_sizes=[500.0, 300.0, 200.0],
        ask_prices=[0.66, 0.67, 0.68],
        ask_sizes=[300.0, 400.0, 250.0],
    )


@pytest.fixture
def sample_market():
    """Create a sample market for testing."""
    return Market(
        market_id="condition-id-123",
        market_slug="test-market",
        question="Will this test pass?",
        asset_type=AssetType.PREDICTION_MARKET,
        exchange=Exchange.POLYMARKET,
        creation_time=datetime.utcnow(),
        metadata={
            "clob_token_ids": ["token-yes-123", "token-no-456"],
            "outcomes": ["Yes", "No"],
        },
    )
```

**Usage:**
```python
class TestTradeValidator:
    def test_valid_trade(self, sample_trade):
        """Test validation of a valid trade."""
        validator = TradeValidator()
        result = validator.validate(sample_trade)
        assert result.valid
        assert len(result.errors) == 0
```

## Mocking

**Framework:** Not currently used - tests use real model instances

**Patterns (for future reference):**
```python
# For async tests, use pytest-asyncio
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result is not None

# For mocking HTTP calls (when needed)
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_with_mocked_http():
    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_get.return_value.__aenter__.return_value.status = 200
        mock_get.return_value.__aenter__.return_value.json = AsyncMock(return_value={...})
        # test code
```

**What to Mock:**
- External API calls (HTTP requests)
- Database connections
- Time-dependent operations
- File system operations

**What NOT to Mock:**
- Data models (Pydantic models are easy to instantiate)
- Pure functions/transformations
- Validators (test their actual logic)

## Test Data

**Factory Pattern (via fixtures):**
```python
@pytest.fixture
def sample_trade():
    """Create a sample trade for testing."""
    return Trade(
        market_id="0xabc123condition",
        token_id="12345678901234567890",
        trade_id="trade-001",
        timestamp=datetime.utcnow(),
        price=0.65,
        size=100.0,
        side=Side.BUY,
        is_buyer_maker=False,
    )
```

**Inline Test Data (for edge cases):**
```python
def test_price_too_low(self):
    trade = Trade(
        market_id="test",
        token_id="test-token",
        trade_id="t1",
        timestamp=datetime.utcnow(),
        price=0.0001,  # Below minimum
        size=100.0,
        side=Side.BUY,
    )
    validator = TradeValidator(min_price=0.001)
    result = validator.validate(trade)
    assert not result.valid
```

**Location:**
- Fixtures in `tests/conftest.py`
- Edge case data inline in test methods
- No separate fixtures directory yet

## Coverage

**Requirements:** No enforced threshold yet

**View Coverage:**
```bash
pytest --cov=tributary                    # Basic coverage
pytest --cov=tributary --cov-report=html  # HTML report
pytest --cov=tributary --cov-report=term-missing  # Show missing lines
```

**Current Coverage Status:**
- `src/tributary/core/models.py`: Well tested
- `src/tributary/validation/validators.py`: Well tested
- Other modules: Limited test coverage

## Test Types

**Unit Tests:**
- Location: `tests/unit/`
- Scope: Individual classes and methods
- Currently tested:
  - `test_models.py`: Trade, OrderBookSnapshot, Market models
  - `test_validators.py`: TradeValidator, OrderBookValidator

**Integration Tests:**
- Location: `tests/integration/`
- Scope: Component interactions (collector + storage, etc.)
- Status: Directory exists but no tests yet

**E2E Tests:**
- Not implemented
- Would test full collection pipeline with real/mock APIs

## Common Patterns

**Testing Computed Properties:**
```python
def test_trade_value_computed(self):
    """Test that value is computed correctly."""
    trade = Trade(
        market_id="market-123",
        token_id="token-123",
        trade_id="trade-001",
        timestamp=datetime.utcnow(),
        price=0.50,
        size=200.0,
        side=Side.SELL,
    )
    assert trade.value == 100.0  # 0.50 * 200
```

**Testing Validation - Valid Input:**
```python
def test_valid_trade(self, sample_trade):
    """Test validation of a valid trade."""
    validator = TradeValidator()
    result = validator.validate(sample_trade)
    assert result.valid
    assert len(result.errors) == 0
```

**Testing Validation - Invalid Input:**
```python
def test_price_too_low(self):
    """Test detection of price below minimum."""
    trade = Trade(
        market_id="test",
        token_id="test-token",
        trade_id="t1",
        timestamp=datetime.utcnow(),
        price=0.0001,  # Below minimum
        size=100.0,
        side=Side.BUY,
    )
    validator = TradeValidator(min_price=0.001)
    result = validator.validate(trade)
    assert not result.valid
    assert any("below minimum" in e for e in result.errors)
```

**Testing Warnings (non-fatal issues):**
```python
def test_old_timestamp_warning(self):
    """Test warning for old timestamp."""
    trade = Trade(
        market_id="test",
        token_id="test-token",
        trade_id="t1",
        timestamp=datetime.utcnow() - timedelta(hours=48),
        price=0.5,
        size=100.0,
        side=Side.BUY,
    )
    validator = TradeValidator(max_age_hours=24)
    result = validator.validate(trade)
    assert result.valid  # Still valid, just warning
    assert any("older than" in w for w in result.warnings)
```

**Testing Floating Point:**
```python
def test_spread_computed(self):
    """Test spread computation."""
    snap = OrderBookSnapshot(...)
    assert snap.spread == 0.10
    assert snap.spread_bps == pytest.approx(2000.0, rel=0.01)
```

**Testing Stateful Behavior:**
```python
def test_duplicate_detection(self):
    """Test duplicate trade detection."""
    validator = TradeValidator()
    trade1 = Trade(trade_id="same-id", ...)
    trade2 = Trade(trade_id="same-id", ...)

    result1 = validator.validate(trade1)
    result2 = validator.validate(trade2)

    assert result1.valid
    assert result2.valid  # Duplicates are valid but warned
    assert any("Duplicate" in w for w in result2.warnings)
```

## Adding New Tests

**For a new model/class:**
1. Create test file: `tests/unit/test_<module>.py`
2. Import the class and dependencies
3. Create test class: `class Test<ClassName>:`
4. Add fixture in `conftest.py` if reusable sample data needed
5. Write tests for:
   - Basic creation/instantiation
   - Computed properties
   - Edge cases (empty, zero, negative values)
   - Validation errors

**For validators:**
1. Test valid input passes
2. Test each validation rule separately
3. Test warnings vs errors distinction
4. Test batch validation if supported

---

*Testing analysis: 2026-01-19*
