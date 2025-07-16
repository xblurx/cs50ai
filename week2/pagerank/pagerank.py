import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")

    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

    ranks = iterate_pagerank(corpus, DAMPING)
    print("PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)

    return pages


def transition_model(
    corpus: dict[str, set[str]], page: str, damping_factor: float
) -> dict[str, float]:
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    links = corpus[page]
    total = len(corpus.keys())

    if not len(links):
        return {page: 1 / total for page in corpus.keys()}

    distrib = {}
    for page in corpus:
        random_jump = (1 - damping_factor) / total
        link_jump = damping_factor / len(links) if page in links else 0
        distrib[page] = random_jump + link_jump

    return distrib


def sample_pagerank(
    corpus: dict[str, set[str]], damping_factor: float, n: int
) -> dict[str, float]:
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = list(corpus.keys())
    counts = {p: 0 for p in pages}
    current = random.choice(pages)

    for _ in range(n):
        counts[current] += 1
        tm = transition_model(corpus, current, damping_factor)
        current = random.choices(
            population=list(tm.keys()), weights=list(tm.values()), k=1
        )[0]

    return {page: counts[page] / n for page in counts}


convergence_threshold = 0.001


def dangling(corpus: dict[str, set[str]], page: str):
    return corpus[page] if len(corpus[page]) else corpus.keys()


def iterate_pagerank(
    corpus: dict[str, set[str]], damping_factor: float
) -> dict[str, float]:
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = list(corpus.keys())
    ranks = {page: 1 / len(pages) for page in pages}

    while True:
        new_ranks = {}

        for page in pages:
            links = {p for p in corpus if page in dangling(corpus, p)}
            new_ranks[page] = (1 - damping_factor) / len(pages) + damping_factor * sum(
                map(
                    lambda page: ranks[page] / len(dangling(corpus, page)),
                    links,
                )
            )

        converged = all(
            [
                abs(new_ranks[page] - ranks[page]) <= convergence_threshold
                for page in pages
            ]
        )

        if converged:
            return new_ranks

        ranks = new_ranks


if __name__ == "__main__":
    main()
