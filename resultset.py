from tabulate import tabulate


class ResultSet:
    def __init__(self, subject_name, items):
        self.subject_name = subject_name

        header1 = "Subject Name"
        header2 = "Distance"
        self.results = [Result(x) for x in items]
        self.table = tabulate(self.results, headers=[header1, header2])

    def print(self):
        print("___"+self.subject_name+"___")
        print(self.table)

    def __getitem__(self, item):
        return self.results[item]

    @property
    def best_result(self):
        return self[0]

    @property
    def match(self):
        return self[1].name[:5] == self.subject_name[:5] and self[1].distance < 2550


class Result:
    def __init__(self, item):
        self.item = item

    def __getitem__(self, item):
        return self.item[item]

    @property
    def name(self):
        return self.item[0]

    @property
    def distance(self):
        return self.item[1]
