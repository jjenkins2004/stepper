"""Step + Stage + Pipeline wired end-to-end on real disk."""

from _helpers import Item


def test_two_stage_pipeline_end_to_end_on_disk(make_pipeline, run):
    p = make_pipeline()
    run(p.run(module="all"))
    base = p.persist_service._base
    # cross-stage dep flows A/prod -> B/consume -> B/note, all landing on disk
    assert (base / "A" / "prod.json").exists()
    assert (base / "B" / "consume.json").exists()
    assert (base / "B" / "note.txt").exists()
    assert p.persist_service.fetch("B/consume", Item) == Item(value=2)
    assert p.persist_service.fetch("B/note", str) == "got 2"


def test_persistence_durable_across_pipeline_instances(make_pipeline, run):
    run(make_pipeline().run(module="a"))          # first instance persists A/prod
    p2 = make_pipeline()                          # fresh instance, same name/run_id -> same dir
    run(p2.run(module="b", step="consume"))       # reads A/prod back off disk
    assert p2.persist_service.fetch("B/consume", Item) == Item(value=2)
