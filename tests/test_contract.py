"""What a node promises versus what it does.

A step's return annotation is the contract for what lands on disk, and a flow's `Flow[X]`
is the contract for what it hands back. These hold both to it — without them a mismatch
serializes to `{}` and surfaces later, in whichever node tried to read it, or never.
"""

import pytest
from _helpers import Item, RootFlow
from pydantic import BaseModel

from stepper import EXIT, START, Flow, depends, edge, step


class Other(BaseModel):
    text: str


# --- a step must return what it declared -----------------------------------------------


def test_returning_the_wrong_model_raises_naming_the_step(run, mem):
    class Liar(Flow[Item]):
        @step
        async def make(self) -> Item:
            return Other(text="oops")  # type: ignore[return-value]

        edges = (edge(START).to(make), edge(make).to(EXIT))

    with pytest.raises(TypeError, match=r"liar/make: declared -> Item but returned Other"):
        run(Liar().run(persist_service=mem))


def test_forgetting_to_return_raises_at_the_step_that_forgot(run, mem):
    """The common slip. Without the check this persists `null` and blows up inside the
    *consumer*, pointing the traceback at the wrong node."""

    class Forgot(Flow[Item]):
        @step
        async def make(self) -> Item:  # type: ignore[return-value]
            pass

        edges = (edge(START).to(make), edge(make).to(EXIT))

    with pytest.raises(TypeError, match="declared -> Item but returned NoneType"):
        run(Forgot().run(persist_service=mem))


def test_a_bad_return_persists_nothing(run, mem):
    class Liar(Flow[Item]):
        @step
        async def make(self) -> Item:
            return Other(text="oops")  # type: ignore[return-value]

        edges = (edge(START).to(make), edge(make).to(EXIT))

    with pytest.raises(TypeError):
        run(Liar().run(persist_service=mem))
    assert mem._store == {}


def test_a_subclass_of_the_declared_model_is_accepted(run, mem):
    class Fancier(Item):
        extra: str = "x"

    class Wider(Flow[Item]):
        @step
        async def make(self) -> Item:
            return Fancier(value=1)

        edges = (edge(START).to(make), edge(make).to(EXIT))

    assert run(Wider().run(persist_service=mem)) == Fancier(value=1)


def test_a_step_that_declares_nothing_may_return_anything(run, mem):
    class Loose(Flow):
        @step
        async def side_effect(self) -> None:
            return "whatever"  # type: ignore[return-value]

        edges = (edge(START).to(side_effect), edge(side_effect).to(EXIT))

    run(Loose().run(persist_service=mem))
    assert mem._store == {"loose/_loop_cursor": mem._store["loose/_loop_cursor"]}


# --- a node is named by the attribute it's bound to -------------------------------------


def _make_check(tag: str):
    """A step factory — the inner function is called `validate` every time, so only the
    attribute name can distinguish the two mounts."""

    @step
    async def validate(self) -> Item:
        return Item(value=len(tag))

    return validate


def test_a_step_is_named_by_its_attribute_not_its_function(run, mem):
    class Factory(Flow[Item]):
        strict = _make_check("strict")
        loose = _make_check("lo")

        @step
        async def total(self, a=depends(strict), b=depends(loose)) -> Item:
            return Item(value=a.value + b.value)

        edges = (
            edge(START).to(strict),
            edge(strict).to(loose),
            edge(loose).to(total),
            edge(total).to(EXIT),
        )

    assert run(Factory().run(persist_service=mem)) == Item(value=8)
    assert "factory/strict" in mem._store and "factory/loose" in mem._store


def test_the_path_a_step_persists_under_is_the_one_depends_reads(mem):
    class Renamed(Flow[Item]):
        elsewhere = _make_check("abc")

        edges = (edge(START).to(elsewhere), edge(elsewhere).to(EXIT))

    root = Renamed().mount(persist_service=mem)
    node = root.node("elsewhere")
    assert node.path == "renamed/elsewhere"
    assert root.key_of(type(root).producers()["elsewhere"]) == "renamed/elsewhere"


# --- composition, not inheritance -------------------------------------------------------


def test_subclassing_a_flow_is_rejected_with_a_useful_message():
    with pytest.raises(TypeError, match="is already a flow. A flow is composed, not inherited"):

        class Tweaked(RootFlow):
            max_steps = 5


def test_subclassing_the_base_flow_is_of_course_fine():
    class Fresh(Flow[Item]):
        @step
        async def only(self) -> Item:
            return Item(value=1)

        edges = (edge(START).to(only), edge(only).to(EXIT))

    assert Fresh.output_model() is Item


# --- settings belong to the run, not to a node -----------------------------------------


def test_settings_passed_to_an_already_mounted_flow_are_rejected(run, mem):
    """Silently dropping a `run_id` would put a run's output on top of another's."""
    root = RootFlow().mount(persist_service=mem)
    with pytest.raises(TypeError, match="settings belong to whatever mounted it"):
        run(root.run(run_id="x"))


def test_running_a_node_of_a_mounted_flow_takes_no_settings(run, mem):
    root = RootFlow().mount(persist_service=mem)
    run(root.run())
    assert run(root.run("left")) == Item(value=2)
