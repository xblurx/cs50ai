from heredity import joint_probability, normalize, update


def test_joint():
    print("f")
    people = {
        "Harry": {"name": "Harry", "mother": "Lily", "father": "James", "trait": None},
        "James": {"name": "James", "mother": None, "father": None, "trait": True},
        "Lily": {"name": "Lily", "mother": None, "father": None, "trait": False},
    }

    res = joint_probability(
        people, one_gene={"Harry"}, two_genes={"James"}, have_trait={"James"}
    )
    print(res)


def test_normalize():
    people = {
        "Harry": {"name": "Harry", "mother": "Lily", "father": "James", "trait": None},
        "James": {"name": "James", "mother": None, "father": None, "trait": True},
        "Lily": {"name": "Lily", "mother": None, "father": None, "trait": False},
    }
    probabilities = {
        person: {"gene": {2: 0, 1: 0, 0: 0}, "trait": {True: 0, False: 0}}
        for person in people
    }

    res = joint_probability(
        people, one_gene={"Harry"}, two_genes={"James"}, have_trait={"James"}
    )
    update(
        probabilities,
        one_gene={"Harry"},
        two_genes={"James"},
        have_trait={"James"},
        p=res,
    )
    normalize(probabilities)

    print(f"probs: {probabilities}")
