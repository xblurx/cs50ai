import sys
from collections import deque
from functools import wraps
from time import time

from crossword import Crossword, Variable


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te - ts))
        return result

    return wrap


type Assignment = dict[Variable, str]


class CrosswordCreator:
    def __init__(self, crossword: Crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy() for var in self.crossword.variables
        }

    def letter_grid(self, assignment: Assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment: Assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment: Assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont

        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size, self.crossword.height * cell_size),
            "black",
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                rect = [
                    (j * cell_size + cell_border, i * cell_size + cell_border),
                    (
                        (j + 1) * cell_size - cell_border,
                        (i + 1) * cell_size - cell_border,
                    ),
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (
                                rect[0][0] + ((interior_size - w) / 2),
                                rect[0][1] + ((interior_size - h) / 2) - 10,
                            ),
                            letters[i][j],
                            fill="black",
                            font=font,
                        )

        img.save(filename)

    @timing
    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def correctly_overlaps(
        self, x: Variable, wordx: str, y: Variable, wordy: str
    ) -> bool | None:
        overlap = self.crossword.overlaps[x, y]

        if overlap:
            i, j = overlap
            return wordx[i] == wordy[j]

        return None

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for variable, domain in self.domains.items():
            self.domains[variable] = {
                word for word in domain if len(word) == variable.length
            }

    def revise(self, x: Variable, y: Variable) -> bool:
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False

        for wordx in set(self.domains[x]):
            satisfies_constraint = any(
                self.correctly_overlaps(x, wordx, y, wordy) for wordy in self.domains[y]
            )

            if not satisfies_constraint:
                self.domains[x].remove(wordx)
                revised = True

        return revised

    def ac3(self, arcs: deque[tuple[Variable, Variable]] | None = None) -> bool:
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        queue = deque(
            arcs
            if arcs is not None
            else [
                (x, y) for x in self.domains.keys() for y in self.crossword.neighbors(x)
            ]
        )

        while len(queue) != 0:
            (x, y) = queue.popleft()
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False

                for z in self.crossword.neighbors(x) - set([y]):
                    queue.append((z, x))

        return True

    def assignment_complete(self, assignment: Assignment) -> bool:
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for variable in self.crossword.variables:
            if variable not in assignment or not assignment[variable]:
                return False

        return True

    def consistent(self, assignment: Assignment) -> bool:
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        words = list(assignment.values())
        if len(words) != len(set(words)):
            return False

        for variable, word in assignment.items():
            if len(word) != variable.length:
                return False

            for neighbor in self.crossword.neighbors(variable):
                val = assignment.get(neighbor)
                if val and not self.correctly_overlaps(variable, word, neighbor, val):
                    return False

        return True

    def order_domain_values1(self, var: Variable, assignment: Assignment) -> list[str]:
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        return [word for word in self.domains[var]]

    def order_domain_values(self, var: Variable, assignment: Assignment) -> list[str]:
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        words = (
            self.domains[var]
            if var not in assignment
            else self.domains[var] - set(assignment[var])
        )

        conflicts = {}
        neighbors = (n for n in self.crossword.neighbors(var) if n not in assignment)

        for w in words:
            count = 0

            for n in neighbors:
                count += sum(
                    [
                        1
                        for y in self.domains[n]
                        if (correct := self.correctly_overlaps(var, w, n, y))
                        is not None
                        and not correct
                    ]
                )

            conflicts[w] = count

        return sorted(
            words,
            key=lambda word: -conflicts[word],
        )

    def select_unassigned_variable1(self, assignment: Assignment) -> Variable:
        for variable in self.crossword.variables:
            if variable not in assignment:
                return variable

        return list(self.crossword.variables)[0]

    def select_unassigned_variable(self, assignment: Assignment) -> Variable:
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        unassigned = (
            pair for pair in self.domains.items() if pair[0] not in assignment
        )

        return min(
            unassigned,
            key=lambda pair: (len(pair[1]), -len(self.crossword.neighbors(pair[0]))),
        )[0]

    def backtrack(self, assignment: Assignment) -> Assignment | None:
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """

        if self.assignment_complete(assignment):
            return assignment

        variable = self.select_unassigned_variable(assignment)

        for value in self.order_domain_values(variable, assignment):
            test_consistency = assignment.copy()
            test_consistency[variable] = value
            if self.consistent(test_consistency):
                assignment[variable] = value
                inferred = self.ac3(
                    deque([(y, variable) for y in self.crossword.neighbors(variable)])
                )
                inferences = [
                    x for x in self.domains.keys() if len(self.domains[x]) == 1
                ]

                if inferred and len(inferences):
                    for x in inferences:
                        assignment[x] = list(self.domains[x])[0]

                result = self.backtrack(assignment)
                if result:
                    return result

                del assignment[variable]

                for x in inferences:
                    del assignment[x]

        return None


def main():
    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
