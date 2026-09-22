import pytest

from conflux.crdts import GCounter, GSet, LWWRegister, ORMap, ORSet, PNCounter, Stamp


def assert_join_laws(a, b, c):
    assert a.merge(b) == b.merge(a)
    assert a.merge(a) == a
    assert a.merge(b).merge(c) == a.merge(b.merge(c))


def test_gcounter_laws():
    assert_join_laws(GCounter({"x": 1}), GCounter({"x": 2, "y": 1}), GCounter({"z": 5}))


def test_gcounter_read():
    a = GCounter().inc("a", 2).inc("b", 3)
    assert a.read() == 5
    assert a.merge(GCounter({"a": 4})).read() == 7


def test_pncounter_laws():
    assert_join_laws(
        PNCounter({"x": (1, 0)}), PNCounter({"x": (0, 2), "y": (1, 1)}), PNCounter({"z": (9, 0)})
    )


def test_pncounter_inc_dec():
    a = PNCounter({"x": (5, 0)})
    b = PNCounter({"x": (0, 2)})
    assert a.merge(b).read() == 3


def test_lww_register_laws():
    s1, s2, s3 = Stamp(1, "a"), Stamp(2, "b"), Stamp(3, "c")
    assert_join_laws(
        LWWRegister("v1", s1), LWWRegister("v2", s2), LWWRegister("v3", s3)
    )


def test_lww_register_newest_wins():
    s1, s2 = Stamp(1, "a"), Stamp(2, "b")
    r = LWWRegister("old", s1).merge(LWWRegister("new", s2))
    assert r.read() == "new"


def test_lww_register_equal_stamp_tiebreak():
    s = Stamp(5, "a")
    assert LWWRegister("low", s).merge(LWWRegister("high", s)).read() == "high"
    assert LWWRegister("high", s).merge(LWWRegister("low", s)).read() == "high"


def test_gset_union():
    a = GSet({"a", "b"})
    b = GSet({"b", "c"})
    assert a.merge(b).read() == frozenset({"a", "b", "c"})


def test_orset_laws():
    a = ORSet({"t1": "x", "t2": "y"}, frozenset({"t1"}))
    b = ORSet({"t3": "z"}, frozenset({"t2"}))
    c = ORSet({"t4": "w"}, frozenset())
    assert_join_laws(a, b, c)


def test_orset_remove_dominates():
    a = ORSet({"t1": "x"}, frozenset())
    r = a.remove("t1")
    assert r.read() == frozenset()
    assert a.merge(r).read() == frozenset()


def test_orset_merge_conflicting_tag_is_deterministic():
    a = ORSet({"t1": "low"}, frozenset())
    b = ORSet({"t1": "high"}, frozenset())
    assert a.merge(b).read() == frozenset({"high"})
    assert b.merge(a).read() == frozenset({"high"})


def test_orset_compact():
    s = ORSet({"t1": "a", "t2": "b"}, frozenset()).remove("t1")
    assert s.read() == frozenset({"b"})
    compacted = s.compact()
    assert compacted.read() == frozenset({"b"})
    assert compacted.elements == {"t2": "b"}
    assert compacted.tombstones == frozenset()


def test_ormap_laws():
    a = ORMap({"x": GCounter({"a": 1})})
    b = ORMap({"y": PNCounter({"b": (1, 1)}), "x": GCounter({"a": 3})})
    c = ORMap({"z": LWWRegister("hi", Stamp(1, "c"))})
    assert_join_laws(a, b, c)


def test_ormap_type_conflict_raises():
    a = ORMap({"x": GCounter({"a": 1})})
    with pytest.raises(TypeError):
        a.put("x", PNCounter({"b": (1, 0)}))


def test_ormap_read():
    root = ORMap({"total": GCounter({"a": 2})}).put("note", LWWRegister("hi", Stamp(1, "a")))
    assert root.read("total") == 2
    assert root.read("note") == "hi"