"""Pipeline — stage lookup, step_map, resolve/run branching, persist root."""

import pytest

from stepper.pipeline import Pipeline

from _helpers import AStage, Item


def test_stage_names_insertion_order(make_pipeline):
    assert make_pipeline().stage_names() == ["a", "b"]


def test_build_stage_uses_pipeline_persist_service(make_pipeline):
    p = make_pipeline()
    stage = p.build_stage("a")
    assert isinstance(stage, AStage)
    assert stage._persist is p.persist_service


def test_build_stage_unknown_raises(make_pipeline):
    with pytest.raises(ValueError, match="has no stage 'zzz'"):
        make_pipeline().build_stage("zzz")


def test_step_map_lists_all_steps_without_running(make_pipeline):
    assert make_pipeline().step_map() == {"a": ["prod"], "b": ["consume", "note"]}


def test_resolve_stage_names(make_pipeline):
    p = make_pipeline()
    assert p._resolve_stage_names("all", None) == ("a", "b")
    assert p._resolve_stage_names("a", None) == ("a",)
    with pytest.raises(ValueError, match="step is not valid with module 'all'"):
        p._resolve_stage_names("all", "prod")


def test_persist_base_dir_is_output_root_name_run_id(make_pipeline, tmp_path):
    p = make_pipeline(name="pp", run_id="rr")
    assert p.persist_service._base == tmp_path / "pp" / "rr"


def test_run_all_runs_every_stage(make_pipeline, run):
    p = make_pipeline()
    run(p.run(module="all"))
    ps = p.persist_service
    assert ps.fetch("A/prod", Item) == Item(value=1)
    assert ps.fetch("B/consume", Item) == Item(value=2)
    assert ps.fetch("B/note", str) == "got 2"


def test_run_single_stage_only(make_pipeline, run):
    p = make_pipeline()
    run(p.run(module="a"))
    assert p.persist_service.fetch("A/prod", Item) == Item(value=1)
    with pytest.raises(FileNotFoundError):
        p.persist_service.fetch("B/consume", Item)  # stage b never ran


def test_run_single_step_only(make_pipeline, run):
    p = make_pipeline()
    run(p.run(module="a"))                       # seed A/prod
    run(p.run(module="b", step="consume"))
    assert p.persist_service.fetch("B/consume", Item) == Item(value=2)
    with pytest.raises(FileNotFoundError):
        p.persist_service.fetch("B/note", str)   # sibling step not run


def test_routing_key_independent_of_stage_name(make_pipeline, run):
    # AStage registered under dict key "renamed"; its stage_name stays "A".
    p = make_pipeline(stages={"renamed": lambda ps: AStage(persist_service=ps)})
    run(p.run(module="renamed"))                 # routing keys off the dict key
    assert p.persist_service.fetch("A/prod", Item) == Item(value=1)  # storage keys off class
    with pytest.raises(ValueError, match="has no stage 'A'"):
        run(p.run(module="A"))


# --- run() returns the last thing it ran (final output, no PersistService poke) ----


def test_run_all_returns_final_stage_results(make_pipeline, run):
    # last stage is B (consume -> Item(2), note -> "got 2"); results in declaration order.
    assert run(make_pipeline().run(module="all")) == [Item(value=2), "got 2"]


def test_run_single_stage_returns_its_results(make_pipeline, run):
    assert run(make_pipeline().run(module="a")) == [Item(value=1)]


def test_run_single_step_returns_that_step_value(make_pipeline, run):
    p = make_pipeline()
    run(p.run(module="a"))                       # seed A/prod for the cross-stage dep
    assert run(p.run(module="b", step="consume")) == Item(value=2)
