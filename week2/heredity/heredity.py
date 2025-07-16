import csv
import itertools
import math
import sys
from typing import TypedDict

PROBS = {
    # Unconditional probabilities for having gene
    "gene": {2: 0.01, 1: 0.03, 0: 0.96},
    "trait": {
        # Probability of trait given two copies of gene
        2: {True: 0.65, False: 0.35},
        # Probability of trait given one copy of gene
        1: {True: 0.56, False: 0.44},
        # Probability of trait given no gene
        0: {True: 0.01, False: 0.99},
    },
    # Mutation probability
    "mutation": 0.01,
}


def main():
    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {"gene": {2: 0, 1: 0, 0: 0}, "trait": {True: 0, False: 0}}
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):
        # Check if current set of people violates known information
        fails_evidence = any(
            (
                people[person]["trait"] is not None
                and people[person]["trait"] != (person in have_trait)
            )
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):
                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (
                    True
                    if row["trait"] == "1"
                    else False
                    if row["trait"] == "0"
                    else None
                ),
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s)
        for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


type Name = str


class Person(TypedDict):
    name: Name
    mother: None | Name
    father: None | Name
    trait: bool


def person_probability(
    person: Name,
    one_gene: set[Name],
    two_genes: set[Name],
    have_trait: set[Name],
):
    count = 1 if person in one_gene else 2 if person in two_genes else 0
    gene = PROBS["gene"][count]
    trait = person in have_trait or PROBS["trait"][count][person in have_trait]

    return gene * trait


def joint_probability(
    people: dict[
        Name,
        Person,
    ],
    one_gene: set[Name],
    two_genes: set[Name],
    have_trait: set[Name],
) -> float:
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """

    def count(x: Name):
        return 1 if x in one_gene else 2 if x in two_genes else 0

    probs = []

    for person in people:
        mother, father = people[person]["mother"], people[person]["father"]

        if not (mother and father):
            i = count(person)
            gene = PROBS["gene"][i]
            trait = PROBS["trait"][i][person in have_trait]

            probs.append(gene * trait)

        else:
            lookup = {0: PROBS["mutation"], 1: 0.5, 2: 1 - PROBS["mutation"]}

            i = count(person)
            m = count(mother)
            f = count(father)
            gene = (
                lookup[m] * (1 - lookup[f]) + (1 - lookup[m]) * lookup[f]
                if i == 1
                else lookup[m] * lookup[f]
                if i == 2
                else (1 - lookup[m]) * (1 - lookup[f])
            )
            trait = PROBS["trait"][i][person in have_trait]

            probs.append(gene * trait)

    return math.prod(list(map(lambda p: round(p, 7), probs)))


class Probability(TypedDict):
    gene: dict[int, float]
    trait: dict[bool, float]


def update(
    probabilities: dict[Name, Probability],
    one_gene: set[Name],
    two_genes: set[Name],
    have_trait: set[Name],
    p: float,
):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        gene = 1 if person in one_gene else 2 if person in two_genes else 0
        probabilities[person]["gene"][gene] += p
        probabilities[person]["trait"][person in have_trait] += p


def normalize(probabilities: dict[Name, Probability]):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for probability_data in probabilities.values():
        for category in probability_data.keys():
            distribution = probability_data[category]
            s = sum(distribution.values())

            if s > 0:
                for p in distribution:
                    distribution[p] /= s


if __name__ == "__main__":
    main()
