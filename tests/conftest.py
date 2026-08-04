"""Shared fixtures.

``run`` drives coroutines without an async plugin. ``persist`` is a real disk-backed
service on an isolated tmp dir, so tests round-trip real values and get the str/bytes/json
split for free. ``mem`` is the in-memory backend, for tests that want to read raw keys.
"""

import asyncio

import pytest

from stepper import DiskPersistService, InMemoryPersistService


@pytest.fixture
def run():
    return asyncio.run


@pytest.fixture
def persist(tmp_path):
    return DiskPersistService(base_dir=tmp_path)


@pytest.fixture
def mem():
    return InMemoryPersistService()
