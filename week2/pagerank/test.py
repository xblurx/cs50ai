import pytest
from pagerank import sample_pagerank, transition_model


def test_simple():
    corpus = {
        "1.html": {"2.html", "3.html"},
        "2.html": {"3.html"},
        "3.html": {"2.html"},
    }
    page = "1.html"
    damping_factor = 0.85

    res = transition_model(corpus, page, damping_factor)

    print(f"transition model : {res}")
    expected = {
        "1.html": 0.05,
        "2.html": 0.475,
        "3.html": 0.475,
    }

    for p in expected:
        assert expected[p] == pytest.approx(res[p], 0.0001)


def test_empty():
    corpus = {
        "1.html": {"2.html", "3.html"},
        "2.html": {},
        "3.html": {"2.html"},
    }
    page = "2.html"
    damping_factor = 0.85

    res = transition_model(corpus, page, damping_factor)

    expected = {
        "1.html": 0.33333,
        "2.html": 0.33333,
        "3.html": 0.33333,
    }

    for p in expected:
        assert expected[p] == pytest.approx(res[p], 0.0001)


def test_sampling():
    corpus = {
        "1.html": {"2.html", "3.html"},
        "2.html": {},
        "3.html": {"2.html"},
    }

    res = sample_pagerank(corpus, 0.85, 10000)

    print(res)

    assert sum([n for n in res.values()]) == 1
