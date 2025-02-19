from argparse import ArgumentTypeError
from typing import TypeVar, Generic
from webbrowser import Error

T = TypeVar('T')

class Cell(Generic[T]):

    def __init__(self, value: T):
        self.Value: T = value

    def __str__(self) -> str:
        return str(self.Value)

    def __repr__(self):
        return str(self)

class Enumerate(Generic[T]):

    def __init__(self, values=None):
        self.__iterate_count: int = 0
        self._values: list[Cell[T]] = [Cell(i) for i in values] if values else []

    def GetValue(self, index: int) -> T:
        assert 0 <= index < len(self._values), "Error, out of bounds"
        return self._values[index].Value

    def SetCell(self, index: int, cell: Cell[T]) -> None:
        assert index <= self.Size
        assert isinstance(cell, Cell)
        self._values[index] = cell

    def AddCell(self, cell: Cell[T]):
        assert isinstance(cell, Cell)
        self._values.append(cell)

    def GetCell(self, index: int) -> Cell[T]:
        return self._values[index]

    @property
    def Size(self):
        return len(self._values)

    def Clone(self) -> 'Enumerate[T]':
        return Enumerate([n.Value for n in self._values])

    def __getitem__(self, index: int | slice) -> 'T | Enumerate[T]':
        if isinstance(index, int):
            return self.GetValue(index)
        elif isinstance(index, slice):
            start: int = index.start or 0
            stop: int = index.stop or len(self._values)
            rep = Enumerate()
            rep._values = [None for _ in range(start, stop)]
            for i in range(start, stop):
                rep._values[i - start] = self._values[i]
            return rep
        else:
            raise ArgumentTypeError("invalid type of data")

    def __setitem__(self, index: int, value: T) -> T:
        assert 0 <= index < len(self._values), "Error, out of bounds"
        assert isinstance(index, int), "index must be integer"
        self._values[index].Value = value
        return value

    def __mul__(self, value: 'float | int | Enumerate[T]'):
        if isinstance(value, (int, float)):
            assert isinstance(value, float) or isinstance(value, int), f"Type error: number must be a number - {value}"
            return Line([n.Value * value for n in self._values])
        elif isinstance(value, Enumerate):
            assert self.Size == value.Size, "Error"
            rep: float = 0
            for i in range(self.Size):
                rep += self.GetValue(i) * value[i]
            return round(rep, 8)
        else:
            raise ArgumentTypeError("Invalid argument type")

    def __add__(self, other: 'Enumerate[T]') -> 'Enumerate[T]':
        assert self.Size == len(other), "Error"
        assert isinstance(other, Enumerate), "Type error"
        return Enumerate([self._values[i].Value + other[i] for i in range(self.Size)])

    def __or__(self, other: 'Enumerate[T]') -> 'Enumerate[T]':
        assert isinstance(other, Enumerate), "Type error"
        rep = Line(self)
        for i in range(len(other)):
            rep._values.append(other._values[i])
        return rep

    def __iter__(self):
        self.__iterate_count = 0
        return self

    def __next__(self) -> T:
        if len(self._values) <= self.__iterate_count:
            raise StopIteration
        rep = self._values[self.__iterate_count].Value
        self.__iterate_count += 1
        return rep

    def __len__(self):
        return self.Size

    def __str__(self) -> str:
        return "| " + ", ".join([str(n) for n in self._values]) + " |"


class Line(Enumerate[T]):

    def __init__(self,  values: list[T] | Enumerate[T] = None):
        if isinstance(values, Enumerate):
            super().__init__([i for i in values])
        else:
            super().__init__(values)


class Colon(Enumerate[T]):

    def __init__(self):
        super().__init__()


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

    def GetValue(self, row: int, column: int) -> T:
        assert isinstance(row, int) and isinstance(column, int)
        assert 0 <= row < self.Height and 0 <= column < self.Width
        return self._lines[row][column]

    def SetValue(self, row: int, column: int, value: T):
        assert 0 <= row < self.Height
        assert 0 <= column < self.Width
        self._lines[row][column] = value

    def GetSlice(self, sl: slice) -> 'Matrix[T]':
        assert isinstance(sl, slice)
        rep = Matrix([])
        for i in range(self.Height):
            rep._lines.append(self.GetLine(i)[sl])
        return rep

    def __getitem__(self, value: tuple[int, int] | slice) -> 'T | Matrix[T]':
        if isinstance(value, tuple):
            return self.GetValue(value[0], value[1])
        elif isinstance(value, slice):
            return self.GetSlice(value)
        raise ArgumentTypeError("Invalid argument")

    def __setitem__(self, position: tuple[int, int], value: T):
        return self.SetValue(position[0], position[1], value)

    def GetLine(self, index: int) -> Line[T]:
        assert 0 <= index < len(self._lines), "Error: out of bounds"
        return self._lines[index]

    def SetLine(self, index: int, line: Line[T]):
        assert 0 <= index < len(self._lines), "Error: out of bounds"
        assert not line in self._lines, "Error: line already exist"
        self._lines[index] = line

    def GetColon(self, index: int):
        assert 0 <= index <= len(self._lines[0]), "Error: out of bounds"
        rep = Colon()
        for line in self._lines:
            rep.AddCell(line.GetCell(index))
        return rep

    def SetColon(self, index: int, colon: Colon[T]):
        assert colon.Size == self.Height
        assert index < self._lines[0].Size
        for i in range(len(colon)):
            self._lines[i].SetCell(index, colon.GetCell(i))

    def Inverse(self) -> 'Matrix[int | float]':
        if self.Height != self.Width:
            return None
        assert isinstance(self._lines[0][0], (float, int))

        clone = self.Clone() | Matrix.MatrixId(self.Height)
        for i in range(self.Height):
            switch = i
            while switch < clone.Height and clone[switch, switch] == 0:
                switch += 1
            if switch == clone.Height:
                switch -= 1
                while switch > 0 and clone[switch, switch] == 0:
                    switch -= 1
            if switch != i:
                tmp = clone.GetLine(i).Clone()
                clone.SetLine(i, clone.GetLine(switch).Clone())
                clone.SetLine(switch, tmp)
        for i in range(self.Width):
            if clone[i, i] == 0:
                raise Error("Inverse matrix doesn't exist")
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

    def MultiplyByValue(self, value: float | int) -> 'Matrix[T]':
        rep = self.Clone()
        for i in range(len(self._lines)):
            rep._lines[i] = rep._lines[i] * value
        return rep

    def MultiplyByMatrix(self, other: 'Matrix[T]') -> 'Matrix[T]':
        assert self.Width == other.Height, "Error: size of matrix"
        rep = Matrix([[0 for _ in range(other.Width)] for _ in range(self.Height)])
        for i in range(self.Height):
            for j in range(other.Width):
                rep[i, j] = self.GetLine(i) * other.GetColon(j)
        return rep

    def __mul__(self, value: 'Matrix[T] | float | int') -> 'Matrix[T]':
        if isinstance(value, Matrix):
            return self.MultiplyByMatrix(value)
        if isinstance(value, float) or isinstance(value, int):
            return self.MultiplyByValue(value)
        raise ArgumentTypeError("Invalid argument type")

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