from typing import TypeVar, Generic

T = TypeVar('T')

class Cell(Generic[T]):

    def __init__(self, value: T):
        self.Value: T = value

    def __str__(self) -> str:
        return str(self.Value)

    def __repr__(self):
        return str(self)

class Enumerate(Generic[T]):

    def __init__(self, values: list[T]):
        self.__count: int = 0
        self._values: list[Cell[T]] = [Cell(i) for i in values]

    def GetValue(self, index: int) -> T:
        assert 0 <= index < len(self._values), "Error, out of bounds"
        return self._values[index].Value

    def __getitem__(self, index: int | slice) -> 'T | Enumerate[T]':
        if isinstance(index, int):
            return self.GetValue(index)
        elif isinstance(index, slice):
            start: int = index.start or 0
            stop: int = index.stop or len(self._values)
            rep = Enumerate([])
            rep._values = [None for _ in range(start, stop)]
            for i in range(start, stop):
                rep._values[i - start] = self._values[i]
            return rep

    def __setitem__(self, index: int, value: T) -> None:
        assert 0 <= index < len(self._values), "Error, out of bounds"
        self._values[index].Value = value

    def __iter__(self):
        self.__count = 0
        return self

    def __mul__(self, value: float | int):
        if isinstance(value, int) or isinstance(value, float):
            assert isinstance(value, float) or isinstance(value, int), f"Type error: number must be a number - {value}"
            return Line([n.Value * value for n in self._values])
        elif isinstance(value, Enumerate):
            assert self.Size == value.Size, "Error"
            rep: float = 0
            for i in range(self.Size):
                rep += self.GetValue(i) * value[i]
            return round(rep, 8)

    def __add__(self, other: 'Enumerate[T]') -> 'Enumerate[T]':
        assert self.Size == len(other), "Error"
        return Enumerate([self._values[i].Value + other[i] for i in range(self.Size)])

    def __or__(self, other: 'Enumerate[T]') -> 'Enumerate[T]':
        rep = Line([0 for _ in range(self.Size + len(other))])
        for i in range(self.Size):
            rep._values[i] = self._values[i]
        for i in range(len(other)):
            rep._values[i + self.Size] = other._values[i]
        return rep

    def __next__(self) -> T:
        if len(self._values) <= self.__count:
            raise StopIteration
        rep = self._values[self.__count].Value
        self.__count += 1
        return rep

    @property
    def Size(self):
        return len(self._values)

    def Clone(self) -> 'Enumerate[T]':
        return Enumerate([n.Value for n in self._values])

    def __len__(self):
        return self.Size

    def __str__(self) -> str:
        return "| " + ", ".join([str(n) for n in self._values]) + " |"

class Line(Enumerate[T]):

    def __init__(self,  values: list[T] | Enumerate[T]):
        if type(values) is list:
            super().__init__(values)
        elif type(values) is Enumerate[T]:
            super().__init__([i for i in values])


class Colon(Enumerate[T]):

    def __init__(self):
        super().__init__([])

    @staticmethod
    def GetColon(lines: list[Line[T]], index: int) -> 'Colon[T]':
        assert index < lines[0].Size
        rep = Colon()
        rep._values = [None for _ in range(len(lines))]
        for i in range(len(lines)):
            rep._values[i] = lines[i]._values[index]
        return rep


class Matrix(Generic[T]):

    def __init__(self, values: list[list[T]]):
        self._lines: list[Line] = [Line(row) for row in values]

    @staticmethod
    def __check_rows(values: list[list[T]]):
        """
        Check if all rows in a 2D list have the same length.

        Parameters:
            values: A 2D list of values.
        Raises:
            ValueError: If any row has a different length than the first row.
        """
        width: int = len(values[0])
        for row in values:
            assert len(row) == width, "All rows must have the same length"

    @staticmethod
    def MatrixNull(n: int) -> 'Matrix[int]':
        return Matrix([0 for _ in range(n)] for _ in range(n))

    @staticmethod
    def MatrixId(n: int) -> 'Matrix[int]':
        return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])

    @property
    def Height(self) -> int:
        """
        :return: Height of matrix
        """
        return len(self._lines)

    @property
    def Width(self):
        """
        :return: Height of matrix
        """
        return len(self._lines[0])

    def __getitem__(self, pos: tuple[int, int] | slice) -> 'T | Matrix[T]':
        if isinstance(pos, tuple):
            row, column = pos
            assert 0 <= row < self.Height
            assert 0 <= column < self.Width
            return self._lines[row][column]
        elif isinstance(pos, slice):
            rep = Matrix([[None] for _ in range(self.Height)])
            for i in range(self.Height):
                rep._lines[i] = self.GetLine(i)[pos]
            return rep

    def __setitem__(self, poss: tuple[int, int], value: T):
        row, column = poss
        assert 0 <= row < self.Height
        assert 0 <= column < self.Width
        self._lines[row][column] = value

    def GetLine(self, index: int) -> Line[T]:
        assert 0 <= index < len(self._lines), "Error: out of bounds"
        return self._lines[index]

    def SetLine(self, index: int, line: Line[T]):
        assert 0 <= index < len(self._lines), "Error: out of bounds"
        assert not line in self._lines, "Error: line already exist"
        self._lines[index] = line

    def GetColon(self, index: int):
        assert 0 <= index <= len(self._lines[0]), "Error: out of bounds"
        return Colon.GetColon(self._lines, index)

    def Inverse(self) -> 'Matrix[int | float]':
        if self.Height != self.Width:
            return None

        clone = self.Clone() | Matrix.MatrixId(self.Height)

        for i in range(self.Height):
            switch = i
            while clone[switch, switch] == 0:
                switch += 1
            if switch != i:
                tmp = clone.GetLine(i).Clone()
                clone.SetLine(i, clone.GetLine(switch).Clone())
                clone.SetLine(switch, tmp)

        for i in range(self.Width):
            cf = 1 / clone[i, i]
            clone.SetLine(i, clone.GetLine(i) * cf)
            for j in range(i + 1, self.Height):
                clone.SetLine(j, clone.GetLine(j) + (clone.GetLine(i) * -clone[j, i]))
        for i in range(self.Width - 1, 0, -1):
            for j in range(i - 1, -1, -1):
                clone.SetLine(j, clone.GetLine(j) + (clone.GetLine(i) * -clone[j, i]))
        return clone[3:]

    def Clone(self):
        return Matrix([[self[i, j] for j in range(self.Width)] for i in range(self.Height)])

    def __mul__(self, value: 'Matrix[T] | float | int') -> 'Matrix[T]':
        if isinstance(value, Matrix):
            assert self.Width == value.Height, "Error: size of matrix"
            rep = Matrix([[0 for _ in range(value.Width)] for _ in range(self.Height)])
            for i in range(self.Height):
                for j in range(value.Width):
                    rep[i, j] = self.GetLine(i) * value.GetColon(j)
            return rep
        if isinstance(value, float) or isinstance(value, int):
            rep = self.Clone()
            for i in range(len(self._lines)):
                rep._lines[i] = rep._lines[i] * value
            return rep

    def __add__(self, other: 'Matrix[T]') -> 'Matrix[T]':
        assert self.Height == other.Height
        assert self.Width == other.Width
        rep = self.Clone()
        for i in range(rep.Height):
            rep._lines[i] = rep._lines[i] + other.GetLine(i)
        return rep

    def __or__(self, other: 'Matrix[T]') -> 'Matrix[T]':
        assert self.Height == other.Height
        assert self.Width == other.Width
        rep = Matrix([[None for _ in range(self.Width * 2)] for _ in range(self.Height)])
        for i in range(self.Height):
            rep.SetLine(i, self.GetLine(i) | other.GetLine(i))
        return rep

    def __str__(self):
        """
        :return: A string representation of the matrix, with each number right-aligned within its column.
        """

        column_widths = [0] * len(self._lines[0])
        for row in self._lines:
            for i, element in enumerate(row):
                column_widths[i] = max(column_widths[i], len(str(element)))

        formatted_string = ""
        for row in self._lines:
            formatted_row = "| "
            for i, element in enumerate(row):
                formatted_row += str(element).rjust(column_widths[i]) + " "
            formatted_string += formatted_row.rstrip() + " |\n"

        return formatted_string

    def __repr__(self):
        return self.__str__()