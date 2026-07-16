"""Shared fixtures for the stepper framework tests.

``run`` drives coroutines without an async plugin. ``persist`` is a real disk-backed
service on an isolated tmp dir (round-trips real values, so tests can assert object
equality and the str/json split for free). ``make_pipeline`` roots pipelines under an
isolated tmp dir via ``output_root=tmp_path`` so runs never touch a repo's ``output/``.
"""

import asyncio

import pytest

from stepper import DiskPersistService, Pipeline


@pytest.fixture
def run():
    return asyncio.run


@pytest.fixture
def persist(tmp_path):
    return DiskPersistService(base_dir=tmp_path)


@pytest.fixture
def make_pipeline(tmp_path):
    """Factory for AStage/BStage pipelines rooted under an isolated tmp dir via
    ``output_root=tmp_path``. Routing keys (``a``/``b``) are deliberately distinct
    from the persist stage_names (``A``/``B``)."""
    from _helpers import AStage, BStage

    def _make(*, name="p", run_id="r1", stages=None):
        return Pipeline(
            name=name,
            run_id=run_id,
            output_root=tmp_path,
            stages=stages
            or {
                "a": lambda ps: AStage(persist_service=ps),
                "b": lambda ps: BStage(persist_service=ps),
            },
        )

    return _make
