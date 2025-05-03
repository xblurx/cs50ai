import random
from typing import Any, List, Set, Tuple, Union


class Minesweeper:
    """
    Minesweeper game representation
    """

    def __init__(self, height: int = 8, width: int = 8, mines: int = 8) -> None:
        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines: Set[Tuple[int, int]] = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found: Set[Tuple[int, int]] = set()

    def print(self) -> None:
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell: Tuple[int, int]) -> int:
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell: Tuple[int, int]) -> int:
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):
                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self) -> bool:
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence:
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells: Set[Tuple[int, int]], count: int) -> None:
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other: Any) -> bool:
        return self.cells == other.cells and self.count == other.count

    def __str__(self) -> str:
        return f"{self.cells} = {self.count}"

    def known_mines(self) -> Set[Tuple[int, int]]:
        """
        Returns the set of all cells in self.cells known to be mines.

        More generally, any time the number of cells is equal to the count,
        we know that all of that sentence’s cells must be mines.
        """
        return self.cells if self.count == len(self.cells) else set()

    def known_safes(self) -> Set[Tuple[int, int]]:
        """
        Returns the set of all cells in self.cells known to be safe.

        Any time we have a sentence whose count is 0,
        we know that all of that sentence’s cells must be safe.
        """
        return self.cells if self.count == 0 else set()

    def mark_mine(self, cell: Tuple[int, int]) -> None:
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.

        C is a mine, we could remove C from the sentence
        and decrease the value of count (since C was a mine
        that contributed to that count)
        """
        if cell in self.cells:
            self.cells.remove(cell)
            if self.count > 0:
                self.count -= 1

    def mark_safe(self, cell: Tuple[int, int]) -> None:
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.

        But if we were told that C were safe,
        we could remove C from the sentence altogether
        """
        if cell in self.cells:
            self.cells.remove(cell)


class MinesweeperAI:
    """
    Minesweeper game player
    """

    def __init__(self, height: int = 8, width: int = 8) -> None:
        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made: Set[Tuple[int, int]] = set()

        # Keep track of cells known to be safe or mines
        self.mines: Set[Tuple[int, int]] = set()
        self.safes: Set[Tuple[int, int]] = set()

        # List of sentences about the game known to be true
        self.knowledge: List[Sentence] = []

    def mark_mine(self, cell: Tuple[int, int]) -> None:
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell: Tuple[int, int]) -> None:
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def neighbours(self, cell: Tuple[int, int]) -> Set[Tuple[int, int]]:
        y, x = cell
        neighbors = set()

        directions = [
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
        ]

        for dx, dy in directions:
            row, col = dy + y, dx + x

            if 0 <= col < self.width and 0 <= row < self.height:
                neighbors.add((row, col))

        return neighbors

    def cleanup(self):
        seen = set()
        result = []

        for s in sorted(
            [s for s in self.knowledge if len(s.cells)], key=lambda s: len(s.cells)
        ):
            key = (frozenset(s.cells), s.count)
            if key not in seen:
                seen.add(key)
                result.append(s)

        return result

    def infer_subsets(self) -> List[Sentence]:
        """
        for each sentence in kb we need to find
        whether a sentence A is a subset of some other sentence B
        if there is such a sentence, then add a new sentence to kb, where:
        {B - A}: countB - countA
        """
        kb = self.cleanup()
        seen = {frozenset(x.cells) for x in kb}

        def generator():
            for s1 in kb:
                for s2 in kb:
                    if not len(s1.cells) or not len(s2.cells):
                        continue

                    if s1.cells.issubset(s2.cells):
                        new_cells = s2.cells - s1.cells

                        if new_cells and frozenset(new_cells) not in seen:
                            yield Sentence(new_cells, s2.count - s1.count)

                    if s2.cells.issubset(s1.cells):
                        new_cells = s1.cells - s2.cells

                        if new_cells and frozenset(new_cells) not in seen:
                            yield Sentence(new_cells, s1.count - s2.count)

        return list(generator())

    def conclusion_loop(self):
        while True:
            self.knowledge = self.cleanup()

            safes = {x for s in self.knowledge for x in s.known_safes()}
            for s in safes:
                self.mark_safe(s)

            mines = {x for s in self.knowledge for x in s.known_mines()}
            for s in mines:
                self.mark_mine(s)

            if not len(safes) and not len(mines):
                break

    def inference_loop(self):
        while True:
            self.conclusion_loop()
            inference = self.infer_subsets()

            if not len(inference):
                break

            self.knowledge.extend(inference)

    def add_knowledge(self, cell: Tuple[int, int], count: int) -> None:
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        self.moves_made.add(cell)

        self.mark_safe(cell)

        neighbours = self.neighbours(cell)
        if neighbours:
            mines = len([x for x in neighbours if x in self.mines])
            sentence = Sentence(
                {
                    x
                    for x in neighbours
                    if x not in self.moves_made
                    and x not in self.safes
                    and x not in self.mines
                },
                count - mines,
            )

            if len(sentence.cells):
                self.knowledge.append(sentence)

        self.inference_loop()

    def make_safe_move(self) -> Union[None, Tuple[int, int]]:
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        for x in self.safes:
            if x not in self.moves_made:
                return x

        return None

    def make_random_move(self) -> Union[None, Tuple[int, int]]:
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        for i in range(0, self.height):
            for j in range(0, self.width):
                if (i, j) not in self.moves_made and (i, j) not in self.mines:
                    return (i, j)

        return None
