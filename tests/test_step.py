"""Step / @step / depends — pure object introspection, no persistence."""

from inspect import iscoroutinefunction

import pytest
from pydantic import BaseModel

from stepper.step import Step, depends, step

from _helpers import AStage, BStage  # real Stage subclasses to use as claim() owners


class Foo(BaseModel):
    x: int


@step
async def make_foo(self) -> Foo:
    return Foo(x=1)


@step
async def no_anno(self): ...  # no return annotation


def test_step_decorator_builds_handle():
    assert isinstance(make_foo, Step)
    assert make_foo.name == "make_foo"
    assert make_foo.model is Foo
    assert make_foo.owner is None
    assert iscoroutinefunction(make_foo.fn)


def test_step_without_return_annotation_has_no_model():
    assert no_anno.model is None


def test_depends_returns_producer_identity():
    assert depends(make_foo) is make_foo


def test_dependencies_maps_params_in_signature_order():
    @step
    async def consumer(self, a=depends(make_foo), b=depends(no_anno)) -> Foo: ...

    deps = consumer.dependencies()
    assert list(deps) == ["a", "b"]
    assert deps == {"a": make_foo, "b": no_anno}


def test_dependencies_empty_when_only_self():
    assert make_foo.dependencies() == {}


def test_dependencies_requires_depends_default():
    @step
    async def bad(self, x) -> Foo: ...  # x not wired with depends()

    with pytest.raises(TypeError, match="param 'x'"):
        bad.dependencies()


def test_claim_sets_owner_and_rejects_second_claim():
    @step
    async def s(self) -> Foo: ...

    s.claim(AStage)
    assert s.owner is AStage
    with pytest.raises(TypeError, match="already belongs"):
        s.claim(BStage)


def test_get_owner_raises_when_unclaimed_then_returns_owner():
    @step
    async def lonely(self) -> Foo: ...

    with pytest.raises(TypeError, match="not claimed"):
        lonely.get_owner()
    lonely.claim(AStage)
    assert lonely.get_owner() is AStage
