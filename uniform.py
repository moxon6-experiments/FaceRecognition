class UniformMapping:
    def __init__(self):
        self.uniform_map = self.get_uniform_mapping(256)

    @staticmethod
    def binary_string_from_integer(x):
        y = bin(x)[2:]
        y = "0"*(8-len(y)) + y
        return y

    def count_switches(self, x):
        y = self.binary_string_from_integer(x)
        z = 0
        for i in range(1, len(y)):
            if y[i] != y[i-1]:
                z += 1
        return z

    def is_uniform(self, x):
        n = self.count_switches(x)
        return n <= 2

    def get_uniform_mapping(self, mapping_range):
        uniforms = [self.is_uniform(x) for x in range(mapping_range)]
        mapping = {}
        map_index = 0
        for i, x in enumerate(uniforms):
            if x:
                mapping[i] = map_index
                map_index += 1
        return mapping

    def __contains__(self, item):
        return item in self.uniform_map

    def __getitem__(self, item):
        return self.uniform_map[item]


uniform_mapping = UniformMapping()


def get_uniform_value(x):
    if x in uniform_mapping:
        return uniform_mapping[x]
    else:
        return 58

