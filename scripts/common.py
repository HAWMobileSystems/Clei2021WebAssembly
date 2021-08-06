import csv
import io
import json
import os
from itertools import groupby
from typing import TypeVar, Iterator, Tuple, Any, List, Union, Dict

import numpy as np
import scipy.stats

DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')

_T = TypeVar('_T')


def get_confidence_error(count: int, mean: float, sem: float, level: float = 0.95) -> float:
    if count < 2:
        return np.NaN
    else:
        return mean - scipy.stats.t.interval(level, count - 1, mean, scale=sem)[0].min()


def groups_to_lists(groups: Iterator[Tuple[Any, Iterator[_T]]]) -> List[List[_T]]:
    result = []
    for k, v in groups:
        result.append(list(it for it in v))
    return result


def avg_list(lst):
    return sum(lst) / len(lst)


class BenchmarkTable:
    benchmark_type: str
    columns: List[str]
    rows: Dict[str, List[Union[int, float]]]

    def __init__(self, benchmark_type):
        self.benchmark_type = benchmark_type
        self.columns = []
        self.rows = {}

    def __str__(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def write_csv(self, dir: str, value_precision: int = None):
        file = os.path.join(dir, self.benchmark_type + '.csv')
        rows = []
        rows.append(['name'] + self.columns)
        for test, values in self.rows.items():
            if value_precision is not None:
                values = list(map(lambda v: '%.*f' % (value_precision, v), values))
            rows.append([test] + values)
        with open(file, 'w') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

    @staticmethod
    def merge(benchmarks: List):
        groups = Benchmark.group_by_name(benchmarks)
        benchmarks = list(map(lambda l: Benchmark.avg(l), groups))
        benchmark_type = benchmarks[0].benchmark_type
        t = BenchmarkTable(benchmark_type)
        t.columns = list(map(lambda b: b.name, benchmarks))
        for benchmark in benchmarks:
            if (benchmark.benchmark_type != benchmark_type):
                raise Exception('benchmark type mismatch')
            for key, value in benchmark.values.items():
                t.rows.setdefault(key, [])
                t.rows[key].append(value)
        return t

    def normalize(self, normal_column_names: List[str]):
        normal_column_indexes = list(map(lambda n: self.columns.index(n), normal_column_names))
        for algorithm, values in self.rows.items():
            normal_value_group = list(map(lambda i: values[i], normal_column_indexes))
            normal = avg_list(normal_value_group)
            self.rows[algorithm] = list(map(lambda v: v / normal * 100, values))

    # @staticmethod
    # def autogenerate(dir: str, benchmark_type: str, value_precision: int = None, normal_column_names: List[str] = None,
    #                  postfix: str = ''):
    #     benchmarks = Benchmark.get_all_benchmarks(dir)
    #     benchmarks = Benchmark.filter_by_type(benchmarks, benchmark_type)
    #     t = BenchmarkTable.merge(benchmarks)
    #     if normal_column_names is not None:
    #         t.normalize(normal_column_names)
    #         t.benchmark_type += postfix
    #     t.write_csv(dir, value_precision)


class Benchmark:
    benchmark_type: str
    name: str
    details: Dict[str, Union[str, int, float]]
    values: Dict[str, Union[int, float]]

    def __init__(self, benchmark_type, name):
        self.benchmark_type = benchmark_type
        self.name = name
        self.details = {}
        self.values = {}

    def __str__(self):
        return self.to_json()

    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def to_csv(self) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['type', 'name', 'test', 'value'])
        for (test, value) in self.values.items():
            writer.writerow([self.benchmark_type, self.name, test, value])
        return output.getvalue()

    def write_json(self, dir: str, overwrite=False):
        os.makedirs(dir, exist_ok=True)
        i = 0
        if overwrite:
            file = os.path.join(dir, "%s-%s.json" % (self.benchmark_type, self.name))
        else:
            while True:
                file = os.path.join(dir, "%s-%s-%d.json" % (self.benchmark_type, self.name, i))
                if not os.path.exists(file):
                    break
                i += 1
        with open(file, 'w') as file:
            file.write(self.to_json())

    def write_csv(self, dir: str, overwrite=False):
        os.makedirs(dir, exist_ok=True)
        i = 0
        if overwrite:
            file = os.path.join(dir, "%s-%s.csv" % (self.benchmark_type, self.name))
        else:
            while True:
                file = os.path.join(dir, "%s-%s-%d.csv" % (self.benchmark_type, self.name, i))
                if not os.path.exists(file):
                    break
                i += 1
        with open(file, 'w') as file:
            file.write(self.to_csv())

    @staticmethod
    def group_by_name(benchmarks: List) -> List[List]:
        return groups_to_lists(groupby(benchmarks, lambda b: b.name))

    @staticmethod
    def group_by_type(benchmarks: List) -> List[List]:
        return groups_to_lists(groupby(benchmarks, lambda b: b.benchmark_type))

    @staticmethod
    def filter_by_type(benchmarks: List, benchmark_type: str) -> List[List]:
        return list(filter(lambda b: b.benchmark_type == benchmark_type, benchmarks))

    @staticmethod
    def avg(benchmarks: List):
        avg_benchmark = Benchmark('', '')
        avg_benchmark.name = benchmarks[0].name
        avg_benchmark.benchmark_type = benchmarks[0].benchmark_type
        count = len(benchmarks)
        for benchmark in benchmarks:
            avg_benchmark.details.update(benchmark.details)
            for key, value in benchmark.values.items():
                avg_benchmark.values[key] = avg_benchmark.values.get(key, 0) + (float(value) / count)
        avg_benchmark.details['avg_of'] = count
        return avg_benchmark

    @staticmethod
    def read_json(path):
        with open(path) as f:
            content = f.read()
            dict = json.loads(content)
            b = Benchmark('', '')
            b.__dict__.update(dict)
            return b

    @staticmethod
    def get_all_benchmarks_from_json(dir: str) -> List:
        files = os.listdir(dir)
        json_files = sorted(list(filter(lambda n: n.endswith('.json'), files)))
        return list(map(lambda b: Benchmark.read_json(os.path.join(dir, b)), json_files))
