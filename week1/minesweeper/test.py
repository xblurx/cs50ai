from copy import deepcopy

from minesweeper import Sentence


def infer_subsets(kb) -> list[Sentence]:
    """
    for each sentence in kb we need to find
    whether a sentence A is a subset of some other sentence B
    if there is such a sentence, then add a new sentence to kb, where:
    {B - A}: countB - countA
    """
    seen = {frozenset(x.cells) for x in kb}

    def generator():
        for i, s1 in enumerate(kb):
            for s2 in kb[i + 1 :]:
                if s1.cells.issubset(s2.cells):
                    new_cells = s2.cells - s1.cells

                    if new_cells and frozenset(new_cells) not in seen:
                        yield Sentence(new_cells, s2.count - s1.count)

                if s2.cells.issubset(s1.cells):
                    new_cells = s1.cells - s2.cells

                    if new_cells and frozenset(new_cells) not in seen:
                        yield Sentence(new_cells, s1.count - s2.count)

    return list(generator())


def inference(kb):
    result = deepcopy(kb)
    seen = set()

    while True:
        inference = [
            s
            for s in infer_subsets(kb)
            if frozenset(s.cells) not in seen and not seen.add(frozenset(s.cells))
        ]

        length = len(result)
        result.extend(inference)
        if length == len(result):
            break

    print("result:")
    for s in sorted(result, key=lambda x: len(x.cells)):
        print(f"{s.cells} = {s.count}")

    return result


def inf(kb):
    cpy = deepcopy(kb)

    while True:
        prev = len(cpy)
        cpy.extend(infer_subsets(cpy))

        if len(cpy) == prev:
            break

    return cpy


def test_complex():
    kb = [
        Sentence({(0, 2), (1, 2)}, 1),
        Sentence({(2, 0), (2, 1)}, 1),
        Sentence({(0, 2), (1, 2), (2, 1), (2, 2), (2, 0)}, 2),
    ]

    inferred = inf(kb)

    for s in sorted(inferred, key=lambda s: len(s.cells)):
        print(f"{s.cells} = {s.count}")

    assert ({(2, 2)}, 0) in [(s.cells, s.count) for s in inferred]
