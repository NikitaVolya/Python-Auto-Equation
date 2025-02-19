from matrix import Matrix

class Parser:

    @staticmethod
    def skip(text: str, condition) -> int:
        i = 0
        while condition(text[i]):
            i += 1
            if len(text) <= i:
                break
        return i

    @staticmethod
    def parse_int(text: str) -> float:
        i = Parser.skip(text, lambda c: c.isdigit() or c == "+" or c == "-")
        if text[:i] == "+" or text[:i] == "":
            return 1
        if text[:i] == "-":
            return -1
        return float(text[:i])

    @staticmethod
    def delete_int(text: str) -> str:
        i = Parser.skip(text, lambda c: c.isdigit() or c == "+" or c == "-")
        return text[i:]

    @staticmethod
    def parce_str(text: str) -> str:
        i = Parser.skip(text, lambda c: c.isalpha())
        return text[:i]

    @staticmethod
    def delete_str(text: str) -> str:
        i = Parser.skip(text, lambda c: c.isalpha())
        return text[i:]

class Program:

    @staticmethod
    def get_letters(data: list[str]) -> [str]:
        rep: set = set()
        tmp = ""
        for line in data:
            for c in line.lower():
                if c == " ":
                    continue
                if c.isalpha():
                    tmp += c
                elif tmp != "":
                    rep.add(tmp)
                    tmp = ""
        return sorted(rep)

    @staticmethod
    def get_matrix(data: list[str], head: list[str]) -> (Matrix[float], Matrix[float]):
        cffs = [[0 for _ in range(len(head))] for _ in range(len(data))]
        reps = [[0] for _ in range(len(data))]
        for i in range(len(data)):
            line, rep = data[i].replace(" ", "").split("=")
            while line != "":
                value = Parser.parse_int(line)
                line = Parser.delete_int(line)
                key = Parser.parce_str(line)
                line = Parser.delete_str(line)
                cffs[i][head.index(key)] = value
            reps[i][0] = float(rep)

        return Matrix(cffs), Matrix(reps)

    @staticmethod
    def translate(equations: list[str]):
        return

    @staticmethod
    def get_result(equations: list[str]) -> None:
        letters = Program.get_letters(equations)

        values, res = Program.get_matrix(equations, letters)
        assert values.Width == res.Height
        rep_matrix = values.Inverse() * res
        for i in range(len(letters)):
            print(letters[i], " = ", rep_matrix[i, 0])

def main():
    equations = [
        "3a + 2b = 5",
        "4b + 2c = 8",
        "3c + a = 2"
    ]
    Program.get_result(equations)


if __name__ == "__main__":
    main()