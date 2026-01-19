"""Transform Polymarket API responses to internal models."""

import json
from datetime import datetime, timezone
from typing import Dict, Any, List

from tributary.core.models import Market, Trade, OrderBookSnapshot, AssetType, Exchange, Side


def _parse_json_field(value: Any, default: Any = None) -> Any:
    """Parse a field that may be a JSON-encoded string or already parsed."""
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value  # Already a list/dict


def transform_market_response(data: Dict[str, Any]) -> Market:
    """Transform Gamma API market response to Market model."""
    # Parse timestamps (always use timezone-aware UTC)
    creation_time = datetime.now(timezone.utc)
    if data.get("createdAt"):
        try:
            creation_time = datetime.fromisoformat(
                data["createdAt"].replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            pass

    resolution_time = None
    if data.get("endDate"):
        try:
            resolution_time = datetime.fromisoformat(
                data["endDate"].replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            pass

    # Extract CLOB token IDs - API returns as JSON-encoded string: "[\"id1\", \"id2\"]"
    raw_token_ids = data.get("clobTokenIds") or data.get("clob_token_ids")
    clob_token_ids = _parse_json_field(raw_token_ids, [])

    # Fallback: try extracting from nested tokens array
    if not clob_token_ids and data.get("tokens"):
        tokens = _parse_json_field(data["tokens"], [])
        for token in tokens:
            if isinstance(token, dict) and token.get("token_id"):
                clob_token_ids.append(token["token_id"])

    # Ensure we have strings with valid token IDs (they are large numbers)
    clob_token_ids = [str(tid) for tid in clob_token_ids if tid and len(str(tid)) > 10]

    # Parse outcomes - API returns as JSON-encoded string: "[\"Yes\", \"No\"]"
    outcomes = _parse_json_field(data.get("outcomes"), [])

    # Parse outcomePrices - API returns as JSON-encoded string: "[\"0.5\", \"0.5\"]"
    outcome_prices = _parse_json_field(data.get("outcomePrices"), [])

    # Extract event information (markets belong to events)
    # Events array contains the parent event(s)
    events = data.get("events", [])
    event_slug = ""
    event_title = ""
    if events and isinstance(events, list) and len(events) > 0:
        event = events[0]
        event_slug = event.get("slug", "")
        event_title = event.get("title", "")

    # For Group Market Events (GMP), groupItemTitle indicates this market's specific outcome
    group_item_title = data.get("groupItemTitle", "")

    return Market(
        market_id=data.get("conditionId", data.get("id", "")),
        market_slug=data.get("slug", ""),
        question=data.get("question", ""),
        asset_type=AssetType.PREDICTION_MARKET,
        exchange=Exchange.POLYMARKET,
        creation_time=creation_time,
        resolution_time=resolution_time,
        metadata={
            "category": data.get("category"),
            "outcomes": outcomes,
            "outcome_prices": outcome_prices,
            "volume": data.get("volume"),
            "liquidity": data.get("liquidity"),
            "clob_token_ids": clob_token_ids,
            "condition_id": data.get("conditionId"),
            "description": data.get("description"),
            "event_slug": event_slug,
            "event_title": event_title,
            "group_item_title": group_item_title,
        },
    )


def transform_trade_response(data: Dict[str, Any], fallback_token_id: str) -> Trade:
    """
    Transform CLOB/Data API trade response to Trade model.

    Args:
        data: Raw trade data from API
        fallback_token_id: Token ID to use if not present in response (for Data API)
    """
    # Parse timestamp - can be various formats (always use timezone-aware UTC)
    timestamp = datetime.now(timezone.utc)
    ts_value = data.get("match_time") or data.get("timestamp") or data.get("createdAt")
    if ts_value:
        if isinstance(ts_value, (int, float)):
            # Unix timestamp in seconds or milliseconds
            if ts_value > 1e12:
                timestamp = datetime.fromtimestamp(ts_value / 1000, tz=timezone.utc)
            else:
                timestamp = datetime.fromtimestamp(ts_value, tz=timezone.utc)
        elif isinstance(ts_value, str):
            try:
                timestamp = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
            except ValueError:
                pass

    # Determine side (API returns "buy"/"sell" or "BUY"/"SELL")
    side_str = data.get("side", "").lower()
    side = Side.BUY if side_str == "buy" else Side.SELL

    # Get price and size
    price = float(data.get("price", 0))
    size = float(data.get("size", 0))

    # Extract IDs:
    # - market_id = conditionId (from 'market' field in CLOB API, 'conditionId' in Data API)
    # - token_id = asset_id (specific outcome token)
    market_id = data.get("market") or data.get("conditionId") or ""
    token_id = data.get("asset_id") or data.get("asset") or fallback_token_id

    # Determine is_buyer_maker from CLOB API fields
    # In CLOB API, 'side' represents the taker's side:
    # - If taker side is "sell", buyer was the maker (resting order)
    # - If taker side is "buy", buyer was the taker (aggressor)
    # Data API doesn't provide this info, so we leave as None (unknown)
    is_buyer_maker = None  # Unknown by default
    trade_type = data.get("type", "").upper()
    if trade_type in ("TAKER", "MAKER"):
        # CLOB API: 'side' is taker's side
        # If taker is selling, the buyer was the maker
        is_buyer_maker = side_str == "sell"

    return Trade(
        market_id=market_id,
        token_id=token_id,
        trade_id=data.get("id") or data.get("trade_id") or str(hash(str(data))),
        timestamp=timestamp,
        price=price,
        size=size,
        side=side,
        is_buyer_maker=is_buyer_maker,
    )


def transform_orderbook_response(data: Dict[str, Any], fallback_token_id: str) -> OrderBookSnapshot:
    """
    Transform CLOB API orderbook response to OrderBookSnapshot model.

    Args:
        data: Raw orderbook data from API
        fallback_token_id: Token ID to use if not present in response
    """
    # Always use timezone-aware UTC
    timestamp = datetime.now(timezone.utc)
    if data.get("timestamp"):
        ts_val = data["timestamp"]
        # Handle both ISO format and unix milliseconds
        if isinstance(ts_val, str) and not ts_val.isdigit():
            try:
                timestamp = datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass
        else:
            # Unix timestamp in milliseconds
            try:
                ts_int = int(ts_val)
                if ts_int > 1e12:
                    timestamp = datetime.fromtimestamp(ts_int / 1000, tz=timezone.utc)
                else:
                    timestamp = datetime.fromtimestamp(ts_int, tz=timezone.utc)
            except (ValueError, TypeError):
                pass

    # Extract IDs from response:
    # - market_id = conditionId (from 'market' field)
    # - token_id = asset_id (specific outcome token)
    market_id = data.get("market", "")
    token_id = data.get("asset_id") or fallback_token_id

    # Parse bids and asks - these are arrays of {price, size} objects
    bids: List[Dict] = data.get("bids", [])
    asks: List[Dict] = data.get("asks", [])

    # Sort bids descending, asks ascending
    bids = sorted(bids, key=lambda x: float(x.get("price", 0)), reverse=True)
    asks = sorted(asks, key=lambda x: float(x.get("price", 0)))

    # Extract price and size arrays
    bid_prices = [float(b.get("price", 0)) for b in bids]
    bid_sizes = [float(b.get("size", 0)) for b in bids]
    ask_prices = [float(a.get("price", 0)) for a in asks]
    ask_sizes = [float(a.get("size", 0)) for a in asks]

    # Best bid/ask
    best_bid = bid_prices[0] if bid_prices else 0.0
    best_ask = ask_prices[0] if ask_prices else 1.0
    bid_size = bid_sizes[0] if bid_sizes else 0.0
    ask_size = ask_sizes[0] if ask_sizes else 0.0

    return OrderBookSnapshot(
        market_id=market_id,
        token_id=token_id,
        timestamp=timestamp,
        best_bid=best_bid,
        best_ask=best_ask,
        bid_size=bid_size,
        ask_size=ask_size,
        bid_prices=bid_prices,
        bid_sizes=bid_sizes,
        ask_prices=ask_prices,
        ask_sizes=ask_sizes,
    )
