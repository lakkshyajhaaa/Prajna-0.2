"""
tests/test_database_manager.py — Prajñā 0.3
pytest tests for core/database_manager.py (non-model functions only)
"""

import os
import json
import tempfile
import shutil

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db_dirs(monkeypatch, tmp_path):
    """Override DB paths to use a temp directory."""
    import core.database_manager as dm
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "stage2"
    stage1_dir.mkdir()
    stage2_dir.mkdir()
    monkeypatch.setattr(dm, "DB_STAGE1_DIR", str(stage1_dir))
    monkeypatch.setattr(dm, "DB_STAGE2_DIR", str(stage2_dir))
    monkeypatch.setattr(dm, "DB_LEGACY_DIR", str(tmp_path))
    return str(stage1_dir), str(stage2_dir)


def _make_fake_embedding(dim: int = 512) -> np.ndarray:
    emb = np.random.randn(1, dim).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return emb


# ---------------------------------------------------------------------------
# load_stage_database
# ---------------------------------------------------------------------------

class TestLoadDatabase:
    def test_empty_directory_returns_empty_dict(self, tmp_db_dirs):
        import core.database_manager as dm
        db = dm.load_stage1_database()
        assert isinstance(db, dict)
        assert len(db) == 0

    def test_loads_saved_embedding(self, tmp_db_dirs):
        import core.database_manager as dm
        stage1_dir, _ = tmp_db_dirs
        person_dir = os.path.join(stage1_dir, "Alice")
        os.makedirs(person_dir, exist_ok=True)
        emb = _make_fake_embedding()
        np.save(os.path.join(person_dir, "embedding.npy"), emb)

        db = dm.load_stage1_database()
        assert "Alice" in db
        assert db["Alice"].shape == (1, 512)

    def test_1d_embedding_reshaped_to_2d(self, tmp_db_dirs):
        import core.database_manager as dm
        stage1_dir, _ = tmp_db_dirs
        person_dir = os.path.join(stage1_dir, "Bob")
        os.makedirs(person_dir, exist_ok=True)
        emb_1d = np.random.randn(512).astype(np.float32)
        np.save(os.path.join(person_dir, "embedding.npy"), emb_1d)

        db = dm.load_stage1_database()
        assert "Bob" in db
        assert db["Bob"].ndim == 2
        assert db["Bob"].shape == (1, 512)

    def test_corrupt_file_skipped(self, tmp_db_dirs):
        import core.database_manager as dm
        stage1_dir, _ = tmp_db_dirs
        person_dir = os.path.join(stage1_dir, "Corrupt")
        os.makedirs(person_dir, exist_ok=True)
        with open(os.path.join(person_dir, "embedding.npy"), "w") as f:
            f.write("not a numpy file")

        # Should not raise; corrupt file is skipped
        db = dm.load_stage1_database()
        assert "Corrupt" not in db


# ---------------------------------------------------------------------------
# search_stage1 / search_stage2
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_returns_sorted_by_similarity(self, tmp_db_dirs):
        import core.database_manager as dm
        stage1_dir, _ = tmp_db_dirs

        # Enroll Alice and Bob with known embeddings
        emb_alice = np.ones((1, 512), dtype=np.float32)
        emb_alice /= np.linalg.norm(emb_alice)
        emb_bob = -np.ones((1, 512), dtype=np.float32)
        emb_bob /= np.linalg.norm(emb_bob)

        for name, emb in [("Alice", emb_alice), ("Bob", emb_bob)]:
            d = os.path.join(stage1_dir, name)
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, "embedding.npy"), emb)

        db = dm.load_stage1_database()
        query = np.ones((1, 512), dtype=np.float32)
        query /= np.linalg.norm(query)

        scores, margin = dm.search_stage1(query, db=db)
        assert scores[0][0] == "Alice"  # Alice is most similar
        assert scores[1][0] == "Bob"
        assert scores[0][1] > scores[1][1]
        assert margin == pytest.approx(scores[0][1] - scores[1][1], abs=1e-5)

    def test_empty_db_returns_empty(self, tmp_db_dirs):
        import core.database_manager as dm
        scores, margin = dm.search_stage1(np.zeros((1, 512)), db={})
        assert scores == []
        assert margin == 0.0


# ---------------------------------------------------------------------------
# update_identity / delete_identity
# ---------------------------------------------------------------------------

class TestUpdateDelete:
    def test_update_overwrites_embedding(self, tmp_db_dirs):
        import core.database_manager as dm
        stage1_dir, stage2_dir = tmp_db_dirs

        emb_old = _make_fake_embedding()
        emb_new = _make_fake_embedding()

        for d in [stage1_dir, stage2_dir]:
            pd = os.path.join(d, "Alice")
            os.makedirs(pd, exist_ok=True)
            np.save(os.path.join(pd, "embedding.npy"), emb_old)

        dm.update_identity("Alice", new_embedding_s1=emb_new, new_embedding_s2=emb_new)

        loaded_s1 = np.load(os.path.join(stage1_dir, "Alice", "embedding.npy"))
        np.testing.assert_allclose(loaded_s1, emb_new, atol=1e-6)

    def test_delete_removes_from_both_stages(self, tmp_db_dirs):
        import core.database_manager as dm
        stage1_dir, stage2_dir = tmp_db_dirs

        for d in [stage1_dir, stage2_dir]:
            pd = os.path.join(d, "Bob")
            os.makedirs(pd, exist_ok=True)
            np.save(os.path.join(pd, "embedding.npy"), _make_fake_embedding())

        result = dm.delete_identity("Bob")
        assert 1 in result["deleted_from_stages"]
        assert 2 in result["deleted_from_stages"]
        assert not os.path.exists(os.path.join(stage1_dir, "Bob"))
        assert not os.path.exists(os.path.join(stage2_dir, "Bob"))

    def test_delete_missing_identity(self, tmp_db_dirs):
        import core.database_manager as dm
        result = dm.delete_identity("NonExistent")
        assert 1 in result["not_found_in_stages"]
        assert 2 in result["not_found_in_stages"]
