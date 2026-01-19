"""QuestDB writer using ILP protocol."""

import json
import logging
from datetime import datetime, timezone
from typing import List, Optional

from questdb.ingress import Sender, TimestampNanos, IngressError

from tributary.core.config import QuestDBConfig
from tributary.core.models import Trade, OrderBookSnapshot, Market
from tributary.core.exceptions import StorageError

logger = logging.getLogger(__name__)


class QuestDBWriter:
    """
    High-performance QuestDB writer using ILP protocol.

    Uses a long-lived questdb.ingress.Sender instance for maximum throughput.
    Per the documentation: "Create longer-lived sender objects, as these are
    not automatically pooled. Instead of creating a new sender object for
    every request, create a single sender object and reuse it."

    Uses HTTP protocol for better error reporting and transaction support.
    Auto-flushing is configured to balance throughput and latency.
    """

    def __init__(self, config: QuestDBConfig, auto_flush_rows: int = 1000):
        """
        Initialize the QuestDB writer.

        Args:
            config: QuestDB configuration
            auto_flush_rows: Flush after this many rows (default 1000)
        """
        self.config = config
        self.auto_flush_rows = auto_flush_rows
        self._sender: Optional[Sender] = None

    def _build_conf_string(self) -> str:
        """Build the ILP configuration string with auto-flush settings."""
        # Use HTTP protocol for better error reporting
        # Configure auto-flush for reasonable batching
        return (
            f"http::addr={self.config.host}:{self.config.http_port};"
            f"auto_flush_rows={self.auto_flush_rows};"
            f"auto_flush_interval=1000;"  # 1 second
        )

    async def connect(self) -> None:
        """Establish a long-lived connection to QuestDB."""
        conf = self._build_conf_string()
        logger.info(f"QuestDB ILP configuration: {conf}")

        try:
            # Create sender with manual lifetime control for long-running service
            self._sender = Sender.from_conf(conf)
            self._sender.establish()
            logger.info(f"Successfully connected to QuestDB at {self.config.host}:{self.config.http_port}")
        except IngressError as e:
            logger.error(f"Failed to connect to QuestDB: {e}")
            logger.error(f"Make sure QuestDB is running: docker compose -f docker/docker-compose.yml up -d")
            self._sender = None
            raise StorageError(f"Failed to connect to QuestDB: {e}")
        except Exception as e:
            import traceback
            logger.error(f"Failed to connect to QuestDB: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self._sender = None
            raise StorageError(f"Failed to connect to QuestDB: {e}")

    @property
    def is_connected(self) -> bool:
        """Check if writer is connected."""
        return self._sender is not None

    async def close(self) -> None:
        """Close the sender connection."""
        if self._sender:
            try:
                self._sender.close()  # Idempotent
                logger.info("QuestDB writer closed")
            except Exception as e:
                logger.warning(f"Error closing QuestDB sender: {e}")
            finally:
                self._sender = None

    async def flush(self) -> None:
        """Explicitly flush pending data to QuestDB."""
        if self._sender:
            try:
                self._sender.flush()
            except IngressError as e:
                logger.error(f"Failed to flush to QuestDB: {e}")
                raise StorageError(f"Failed to flush: {e}")

    async def write_trades(self, trades: List[Trade]) -> int:
        """
        Write trades to QuestDB.

        Args:
            trades: List of trades to write

        Returns:
            Number of trades written
        """
        if not self.is_connected:
            raise StorageError("Writer not connected")

        count = 0
        for trade in trades:
            try:
                # Build columns dict, only include is_buyer_maker if known
                columns = {
                    "trade_id": trade.trade_id,
                    "price": trade.price,
                    "size": trade.size,
                    "value": trade.value,
                }
                if trade.is_buyer_maker is not None:
                    columns["is_buyer_maker"] = trade.is_buyer_maker

                self._sender.row(
                    "trades",
                    symbols={
                        "market_id": trade.market_id,
                        "token_id": trade.token_id,
                        "side": trade.side.value,
                    },
                    columns=columns,
                    at=TimestampNanos.from_datetime(trade.timestamp),
                )
                count += 1
            except IngressError as e:
                logger.error(f"Failed to write trade {trade.trade_id}: {e}")

        # Auto-flush is enabled, but flush explicitly for small batches
        if count > 0 and count < self.auto_flush_rows:
            await self.flush()

        return count

    async def write_orderbook_snapshots(self, snapshots: List[OrderBookSnapshot]) -> int:
        """
        Write orderbook snapshots to QuestDB.

        Args:
            snapshots: List of snapshots to write

        Returns:
            Number of snapshots written
        """
        if not self.is_connected:
            raise StorageError("Writer not connected")

        count = 0
        for snap in snapshots:
            try:
                self._sender.row(
                    "orderbook_snapshots",
                    symbols={
                        "market_id": snap.market_id,
                        "token_id": snap.token_id,
                    },
                    columns={
                        "best_bid": snap.best_bid,
                        "best_ask": snap.best_ask,
                        "bid_size": snap.bid_size,
                        "ask_size": snap.ask_size,
                        "mid_price": snap.mid_price,
                        "spread": snap.spread,
                        "spread_bps": snap.spread_bps,
                        "bid_prices": json.dumps(snap.bid_prices),
                        "bid_sizes": json.dumps(snap.bid_sizes),
                        "ask_prices": json.dumps(snap.ask_prices),
                        "ask_sizes": json.dumps(snap.ask_sizes),
                        "total_bid_volume": snap.total_bid_volume,
                        "total_ask_volume": snap.total_ask_volume,
                    },
                    at=TimestampNanos.from_datetime(snap.timestamp),
                )
                count += 1
            except IngressError as e:
                logger.error(f"Failed to write snapshot for {snap.market_id}: {e}")

        # Flush orderbook snapshots immediately for real-time visibility
        if count > 0:
            await self.flush()

        return count

    async def write_market(self, market: Market) -> None:
        """
        Write or update market metadata.

        Args:
            market: Market to write
        """
        if not self.is_connected:
            raise StorageError("Writer not connected")

        try:
            # Extract event info from metadata for easier querying
            event_slug = market.metadata.get("event_slug", "")
            event_title = market.metadata.get("event_title", "")
            group_item_title = market.metadata.get("group_item_title", "")

            # Build columns dict - only include resolution_time if not None
            columns = {
                "question": market.question[:1000] if market.question else "",
                "event_title": event_title[:500] if event_title else "",
                "group_item_title": group_item_title[:200] if group_item_title else "",
                "metadata": json.dumps(market.metadata),
            }

            # Add resolution_time as TimestampNanos if present (ILP requires timestamp type, not string)
            if market.resolution_time:
                columns["resolution_time"] = TimestampNanos.from_datetime(market.resolution_time)

            self._sender.row(
                "markets",
                symbols={
                    "market_id": market.market_id,
                    "market_slug": market.market_slug,
                    "event_slug": event_slug if event_slug else market.market_slug,
                    "asset_type": market.asset_type.value,
                    "exchange": market.exchange.value,
                },
                columns=columns,
                at=TimestampNanos.from_datetime(market.creation_time),
            )
            # Flush immediately for market metadata
            await self.flush()
        except IngressError as e:
            raise StorageError(f"Failed to write market {market.market_id}: {e}")

    async def write_metric(
        self,
        collector: str,
        market_id: str,
        metric_name: str,
        metric_value: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Write a collection metric.

        Args:
            collector: Collector name
            market_id: Market ID
            metric_name: Name of the metric
            metric_value: Value of the metric
            timestamp: Optional timestamp (defaults to now)
        """
        if not self.is_connected:
            return

        ts = timestamp or datetime.now(timezone.utc)

        try:
            self._sender.row(
                "collection_metrics",
                symbols={
                    "collector": collector,
                    "market_id": market_id,
                    "metric_name": metric_name,
                },
                columns={
                    "metric_value": metric_value,
                },
                at=TimestampNanos.from_datetime(ts),
            )
            # Metrics use auto-flush, no explicit flush needed
        except IngressError as e:
            logger.error(f"Failed to write metric: {e}")
