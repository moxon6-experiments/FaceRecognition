from tabulate import tabulate


class ResultSet:
    def __init__(self, subject_name, items, num_results):
        self.subject_name = subject_name
        self.num_result = num_results

        header1 = "Subject Name"
        header2 = "Distance"
        self.results = [Result(x) for x in items]

        # Lower than 1 is incredibly low, therefore likely same image
        if self.results[0].distance < 0.5:
            self.results = self.results[1:]

        if num_results is not None:
            results = self.results[:num_results]
        else:
            results = self.results

        self.table = tabulate(results, headers=[header1, header2])

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
        return self[0].name[:5] == self.subject_name[:5]


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
