from collections import Counter

from clwe_chains.routes import aus_chains


def test_aus_chains_count_and_shape():
    routes = aus_chains("en", "ar", ["es", "ga", "mt"])
    assert len(routes) == 15
    assert Counter(len(r) for r in routes) == {3: 3, 4: 6, 5: 6}
    assert all(r[0] == "en" and r[-1] == "ar" and len(set(r)) == len(r) for r in routes)
    assert routes[0] == ["en", "es", "ar"]
    assert len({tuple(r) for r in routes}) == 15
