"""Running a mounted flow: paths, keys, what a flow returns, and addressing."""

import pytest
from _helpers import DoubleFlow, Item, RootFlow
from pydantic import BaseModel

from stepper import EXIT, START, Flow, InMemoryPersistService, Step, depends, edge, require, step


class Note(BaseModel):
    text: str


# --- paths and keys -------------------------------------------------------------------


def test_paths_are_the_chain_of_names_from_the_root(mem):
    root = RootFlow().mount(persist_service=mem)
    assert root.path == "root"
    assert root.node("start").path == "root/start"
    assert root.node("left").path == "root/left"
    assert root.node("left/double").path == "root/left/double"


def test_two_mounts_of_one_class_get_separate_keys(run, mem):
    run(RootFlow().run(persist_service=mem))
    assert sorted(mem._store) == [
        "root",
        "root/_loop_cursor",
        "root/left",
        "root/left/_loop_cursor",
        "root/left/double",
        "root/right",
        "root/right/_loop_cursor",
        "root/right/double",
        "root/start",
        "root/total",
    ]


def test_run_id_prefixes_keys_but_not_paths(run, mem):
    root = RootFlow().mount(run_id="r7", persist_service=mem)
    run(root.run())
    assert all(k.startswith("r7/") for k in mem._store)
    assert root.node("left/double").path == "root/left/double"


def test_a_flow_persists_its_own_value_under_its_own_path(run, mem):
    run(RootFlow().run(persist_service=mem))
    # left doubles 1 -> 2, and its flow value is what its terminal produced.
    assert mem.fetch("root/left", Item) == Item(value=2)
    assert mem.fetch("root/left/double", Item) == Item(value=2)


def test_disk_keys_land_under_base_dir(run, tmp_path, persist):
    run(RootFlow().run(run_id="r1", persist_service=persist))
    assert (tmp_path / "r1" / "root" / "start.json").exists()
    assert (tmp_path / "r1" / "root" / "left" / "double.json").exists()
    assert (tmp_path / "r1" / "root.json").exists()


# --- values ---------------------------------------------------------------------------


def test_a_run_returns_the_terminals_value(run, mem):
    # start=1 -> left=2 -> right=4 -> total=6
    assert run(RootFlow().run(persist_service=mem)) == Item(value=6)


def test_depends_on_a_flow_reads_that_flows_value(run, mem):
    run(RootFlow().run(persist_service=mem))
    assert mem.fetch("root/total", Item) == Item(value=6)


def test_a_step_with_no_annotation_persists_nothing_and_passes_none(run, mem):
    seen: list[object] = []

    class Quiet(Flow[Item]):
        @step
        async def side_effect(self) -> None:
            return None

        @step
        async def after(self, nothing=depends(side_effect)) -> Item:
            seen.append(nothing)
            return Item(value=1)

        edges = (edge(START).to(side_effect), edge(side_effect).to(after), edge(after).to(EXIT))

    run(Quiet().run(persist_service=mem))
    assert seen == [None]
    assert "quiet/side_effect" not in mem._store


def test_a_str_output_round_trips(run, tmp_path, persist):
    class Text(Flow[str]):
        @step
        async def line(self) -> str:
            return "hello"

        edges = (edge(START).to(line), edge(line).to(EXIT))

    assert run(Text().run(persist_service=persist)) == "hello"
    assert (tmp_path / "text" / "line.txt").read_text() == "hello"


# --- addressing -----------------------------------------------------------------------


def test_running_one_step_by_path(run, mem):
    root = RootFlow().mount(persist_service=mem)
    run(root.run())
    assert run(root.run("left/double")) == Item(value=2)


def test_running_one_subtree_by_path(run, mem):
    root = RootFlow().mount(persist_service=mem)
    run(root.run())
    assert run(root.run("right")) == Item(value=4)


def test_an_unknown_segment_raises(run, mem):
    root = RootFlow().mount(persist_service=mem)
    with pytest.raises(ValueError, match="no node 'nope'"):
        run(root.run("nope"))


def test_addressing_inside_a_step_raises(run, mem):
    root = RootFlow().mount(persist_service=mem)
    with pytest.raises(ValueError, match="a step is a leaf"):
        run(root.run("start/deeper"))


def test_node_resolves_the_same_paths_run_does(mem):
    root = RootFlow().mount(persist_service=mem)
    assert isinstance(root.node("left/double"), Step)
    assert isinstance(root.node("left"), Flow)
    with pytest.raises(ValueError, match="no node 'nope'"):
        root.node("nope")


def test_a_step_run_alone_reads_what_is_persisted(run, mem):
    root = RootFlow().mount(persist_service=mem)
    with pytest.raises(FileNotFoundError):
        run(root.run("left/double"))       # nothing has produced `start` yet


def test_node_addressing_inside_a_step_raises():
    root = RootFlow().mount(persist_service=InMemoryPersistService())
    with pytest.raises(ValueError, match="a step is a leaf"):
        root.node("start/deeper")


# --- describe -------------------------------------------------------------------------


def test_describe_reports_the_mounted_tree(mem):
    tree = RootFlow().mount(persist_service=mem).describe()
    assert tree["path"] == "root"
    assert tree["flow"] == "RootFlow"
    assert tree["output"] == "Item"
    assert tree["requires"] == {}
    assert tree["terminals"] == ["total"]
    # Declaration order, exactly as the class body reads.
    assert [n["path"] for n in tree["nodes"]] == [
        "root/start",
        "root/left",
        "root/right",
        "root/total",
    ]


def test_describe_reports_a_childs_requirements(mem):
    tree = RootFlow().mount(persist_service=mem).describe()
    left = next(n for n in tree["nodes"] if n["path"] == "root/left")
    assert left["requires"] == {"seed": "Item"}
    assert left["nodes"] == [{"path": "root/left/double", "kind": "step", "output": "Item"}]


# --- open and closed ------------------------------------------------------------------


def test_an_open_flow_cannot_be_a_root(run, mem):
    with pytest.raises(TypeError, match="requires seed, so it can't be the root"):
        run(DoubleFlow().run(persist_service=mem))


def test_a_wrapper_closes_an_open_flow(run, mem):
    class Harness(Flow[Item]):
        @step
        async def seed(self) -> Item:
            return Item(value=5)

        inner = DoubleFlow.bind(seed=seed)

        edges = (edge(START).to(seed), edge(seed).to(inner), edge(inner).to(EXIT))

    assert run(Harness().run(persist_service=mem)) == Item(value=10)


def test_mounting_is_idempotent(mem):
    root = RootFlow()
    assert root.mount(persist_service=mem) is root.mount() is root


def test_an_unmounted_node_says_so():
    with pytest.raises(RuntimeError, match="not mounted"):
        RootFlow().ctx


# --- requirements resolve outward -----------------------------------------------------


def test_a_requirement_forwarded_two_levels_resolves_to_the_original(run, mem):
    class Middle(Flow[Item]):
        seed = require(Item)
        inner = DoubleFlow.bind(seed=seed)

        edges = (edge(START).to(inner), edge(inner).to(EXIT))

    class Outer(Flow[Item]):
        @step
        async def origin(self) -> Item:
            return Item(value=3)

        mid = Middle.bind(seed=origin)

        edges = (edge(START).to(origin), edge(origin).to(mid), edge(mid).to(EXIT))

    root = Outer().mount(persist_service=mem)
    inner = root.node("mid/inner")
    assert isinstance(inner, Flow)
    # The chain of bindings lands on the one step that actually produces the value.
    assert inner.key_of(type(inner).plan().requirements["seed"]) == "outer/origin"
    assert run(root.run()) == Item(value=6)
