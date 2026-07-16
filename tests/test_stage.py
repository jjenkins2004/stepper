"""Stage — class-creation validation and runner behavior (fetch -> run -> persist)."""

import pytest

from stepper.stage import Stage
from stepper.step import Step, depends, step

from _helpers import AStage, BStage, Item


# --- class-creation validation (__init_subclass__ / _check_steps) ---

def test_missing_steps_raises():
    with pytest.raises(TypeError, match="must declare"):

        class NoSteps(Stage): ...


def test_non_step_entry_raises():
    with pytest.raises(TypeError, match="must declare"):

        class Bad(Stage):
            steps = ("not-a-step",)  # pyright: ignore  # intentionally not a Step


def test_duplicate_same_object_raises():
    @step
    async def a(self) -> Item: ...

    with pytest.raises(TypeError, match="duplicate"):

        class Dup(Stage):
            steps = (a, a)


def test_name_collision_distinct_objects_collapse(persist):
    """Dedup is identity-based (set()), but runners key by .name — two distinct
    Steps sharing a name pass validation yet silently collapse to one."""

    @step
    async def a(self) -> Item: ...

    twin = Step(a.fn)  # distinct Step object, same .name

    class Collide(Stage):
        steps = (a, twin)  # does NOT raise

    assert Collide(persist_service=persist).get_steps() == ["a"]


def test_stage_name_inferred_from_class_name():
    @step
    async def a(self) -> Item: ...

    class ExtractStage(Stage):
        steps = (a,)

    @step
    async def b(self) -> Item: ...

    class Plain(Stage):  # no "Stage" suffix -> name unchanged
        steps = (b,)

    assert (ExtractStage.stage_name, Plain.stage_name) == ("Extract", "Plain")


def test_step_claimed_by_two_stages_raises():
    @step
    async def shared(self) -> Item: ...

    class First(Stage):
        steps = (shared,)

    assert shared.owner is First
    with pytest.raises(TypeError, match="already belongs"):

        class Second(Stage):
            steps = (shared,)


# --- runner behavior (real DiskPersistService via `persist` fixture) ---

def test_get_steps_declared_order(persist):
    assert BStage(persist_service=persist).get_steps() == ["consume", "note"]


def test_run_steps_runs_in_order_and_persists(persist, tmp_path, run):
    results = run(AStage(persist_service=persist).run_steps())
    assert results == [Item(value=1)]
    assert persist.fetch("A/prod", Item) == Item(value=1)
    assert (tmp_path / "A" / "prod.json").exists()


def test_deps_feed_persisted_values_and_str_json_split(persist, tmp_path, run):
    persist.persist("A/prod", Item(value=1), Item)  # seed cross-stage producer
    results = run(BStage(persist_service=persist).run_steps())
    # cross-stage dep (A/prod) -> consume, same-stage dep (B/consume) -> note
    assert results == [Item(value=2), "got 2"]
    assert persist.fetch("B/consume", Item) == Item(value=2)
    assert persist.fetch("B/note", str) == "got 2"
    assert (tmp_path / "B" / "consume.json").exists()  # model -> .json
    assert (tmp_path / "B" / "note.txt").exists()      # str   -> .txt


def test_model_none_step_is_not_persisted(persist, tmp_path, run):
    @step
    async def noop(self):  # no return annotation -> model is None
        return "ignored"

    class SilentStage(Stage):
        steps = (noop,)

    run(SilentStage(persist_service=persist).run_step("noop"))
    assert not (tmp_path / "Silent").exists()


def test_run_step_returns_value_and_reads_seeded_dep(persist, run):
    persist.persist("A/prod", Item(value=5), Item)
    assert run(BStage(persist_service=persist).run_step("consume")) == Item(value=6)


def test_run_step_unknown_raises(persist, run):
    with pytest.raises(ValueError, match="Unknown step: nope"):
        run(AStage(persist_service=persist).run_step("nope"))


def test_step_body_exception_propagates_and_persists_nothing(persist, tmp_path, run):
    @step
    async def boom(self) -> Item:  # annotated so no-persist assertion isn't vacuous
        raise RuntimeError("kaboom")

    class BoomStage(Stage):
        steps = (boom,)

    with pytest.raises(RuntimeError, match="kaboom"):
        run(BoomStage(persist_service=persist).run_step("boom"))
    assert not (tmp_path / "Boom").exists()


def test_unseeded_dependency_raises_file_not_found(persist, run):
    with pytest.raises(FileNotFoundError):
        run(BStage(persist_service=persist).run_step("consume"))


def test_run_steps_partial_completion_persists_completed_only(persist, tmp_path, run):
    """run_steps (default, no fail_fast): a failing branch skips its dependents but
    independent steps complete and stay on disk — clean partial completion, no raise."""

    @step
    async def ok(self) -> Item:
        return Item(value=1)

    @step
    async def boom(self) -> Item:  # annotated so the "file absent" assertion isn't vacuous
        raise RuntimeError("kaboom")

    @step
    async def after_boom(self, up=depends(boom)) -> Item:
        return Item(value=2)

    class PartialStage(Stage):
        steps = (ok, boom, after_boom)

    results = run(PartialStage(persist_service=persist).run_steps())

    assert results == [Item(value=1)]                              # only ok completed
    assert persist.fetch("Partial/ok", Item) == Item(value=1)
    assert (tmp_path / "Partial" / "ok.json").exists()
    assert not (tmp_path / "Partial" / "boom.json").exists()       # failed -> not persisted
    assert not (tmp_path / "Partial" / "after_boom.json").exists()  # skipped -> not persisted
