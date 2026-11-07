"""Unit tests for QueueManager tier classification and atomic claims."""

import pytest
from pathlib import Path

from topaz_cluster.core.queue_manager import JobItem, QueueManager


def test_job_item_tier_classification():
    j_464 = JobItem(filename="test_464.mp4", stem="test_464", width=640, height=464, tier="")
    assert j_464.tier == "464p"

    j_480 = JobItem(filename="test_480.mp4", stem="test_480", width=720, height=480, tier="")
    assert j_480.tier == "480p"

    j_720 = JobItem(filename="test_720.mp4", stem="test_720", width=1280, height=720, tier="")
    assert j_720.tier == "720p"

    j_1080 = JobItem(filename="test_1080.mp4", stem="test_1080", width=1920, height=1080, tier="")
    assert j_1080.tier == "1080p"


def test_build_node_queue_prioritization():
    qm = QueueManager(
        archive_host="test_host",
        source_dir="/test/src",
        destination_dir="/test/dst",
        claims_dir=Path("/tmp/claims_test"),
    )

    j1 = JobItem("a_464.mp4", "a_464", 640, 464, "")
    j2 = JobItem("b_464.mp4", "b_464", 640, 464, "")
    j3 = JobItem("c_720.mp4", "c_720", 1280, 720, "")
    j4 = JobItem("d_1080.mp4", "d_1080", 1920, 1080, "")
    all_jobs = [j1, j2, j3, j4]

    # Node prioritizing 464p reversed, then 720p
    q = qm.build_node_queue(["464p_reverse", "720p"], all_jobs)
    assert len(q) == 4
    # Reversed 464p should place j2 before j1
    assert q[0].stem == "b_464"
    assert q[1].stem == "a_464"
    assert q[2].stem == "c_720"
    assert q[3].stem == "d_1080"


def test_atomic_claim_lease_lifecycle(tmp_path):
    claims_dir = tmp_path / "claimed_stems"
    qm = QueueManager(
        archive_host="test_host",
        source_dir="/test/src",
        destination_dir="/test/dst",
        claims_dir=claims_dir,
    )

    stem = "test_video_123"

    # First claim succeeds
    assert qm.try_claim(stem, worker_tag="rtx5080") is True
    assert (claims_dir / f"{stem}.lock").exists()

    # Second claim for same stem fails (already leased)
    assert qm.try_claim(stem, worker_tag="v100") is False

    # Check active claims
    active = qm.fetch_active_claims()
    assert stem in active

    # Release claim
    qm.release_claim(stem)
    assert not (claims_dir / f"{stem}.lock").exists()

    # Now another worker can claim it
    assert qm.try_claim(stem, worker_tag="v100") is True
