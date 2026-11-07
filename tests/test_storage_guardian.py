"""Unit tests for StorageGuardian disk capacity gating."""

import threading
import time
import pytest

from topaz_cluster.core.storage_guardian import StorageGuardian


def test_storage_guardian_threshold_logic(monkeypatch):
    guardian = StorageGuardian(
        archive_host="user@archive-host",
        destination_dir="/srv/archive/output",
        min_free_gb=25.0,
        cache_ttl_sec=0.0,
    )

    # Mock query_free_space_gb returning 50.0 GB (Safe)
    monkeypatch.setattr(guardian, "query_free_space_gb", lambda: 50.0)
    safe, free_gb = guardian.is_safe()
    assert safe is True
    assert free_gb == 50.0

    # Mock query_free_space_gb returning 10.0 GB (Unsafe / Gated)
    monkeypatch.setattr(guardian, "query_free_space_gb", lambda: 10.0)
    safe, free_gb = guardian.is_safe()
    assert safe is False
    assert free_gb == 10.0

    # Custom threshold override
    safe_custom, _ = guardian.is_safe(min_gb=5.0)
    assert safe_custom is True


def test_storage_guardian_caching_ttl(monkeypatch):
    guardian = StorageGuardian(
        archive_host="user@archive-host",
        destination_dir="/srv/archive/output",
        min_free_gb=25.0,
        cache_ttl_sec=10.0,
    )

    call_count = 0

    def mock_query():
        nonlocal call_count
        call_count += 1
        return 100.0

    # Manually populate cache
    guardian._cached_free_gb = 100.0
    guardian._last_check_epoch = time.time()

    monkeypatch.setattr(guardian, "query_free_space_gb", mock_query)

    # Calling is_safe should use mock_query
    guardian.is_safe()
    assert call_count == 1


def test_storage_guardian_stop_event_breakout(monkeypatch):
    guardian = StorageGuardian(
        archive_host="user@archive-host",
        destination_dir="/srv/archive/output",
        min_free_gb=25.0,
    )
    # Always unsafe
    monkeypatch.setattr(guardian, "is_safe", lambda: (False, 5.0))

    stop_event = threading.Event()
    stop_event.set()

    # Must return immediately when stop_event is set without looping
    t0 = time.time()
    res = guardian.wait_until_safe(worker_name="TestWorker", poll_interval_sec=10, stop_event=stop_event)
    assert time.time() - t0 < 1.0
    assert res == 0.0
