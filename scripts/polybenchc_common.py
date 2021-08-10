import gzip
import os
import shutil
from time import perf_counter
from typing import List

from scipy import stats
import pandas as pd
from pandas import DataFrame, Series

from common import Benchmark, get_confidence_error
from config import POLYBENCHC_PATH, CLANG_BIN_PATH, WASI_LIBC_PATH, NODEJS_BIN_PATH, \
    SPIDERMONKEY_BIN_PATH, WASMTIME_BIN_PATH, WAVM_BIN_PATH, EMCC_BIN_PATH, WASMER_BIN_PATH

CWD = os.path.dirname(os.path.realpath(__file__))
RESULT_DIR = os.path.join(CWD, '..', 'data', 'polybenchc')
MEASUREMENTS_DIR = os.path.join(RESULT_DIR, 'measurements')
ANALYSIS_DIR = os.path.join(RESULT_DIR, 'analysis')
TMP_DIR = os.path.join(CWD, 'tmp', 'polybenchc')
BIN_DIR = os.path.join(TMP_DIR, 'bin')

POLYBENCHC_BENCHMARK_DIRS = [
    os.path.join(POLYBENCHC_PATH, 'datamining'),
    os.path.join(POLYBENCHC_PATH, 'linear-algebra'),
    os.path.join(POLYBENCHC_PATH, 'medley'),
    os.path.join(POLYBENCHC_PATH, 'stencils')
]

COMPILE_TARGET_X86_64_GENERIC = 'x8664generic'
COMPILE_TARGET_X86_64_NATIVE = 'x8664native'
COMPILE_TARGET_WASM = 'wasm'
COMPILE_TARGET_WASM_SIMD = 'wasmsimd'
COMPILE_TARGET_JS = 'js'
COMPILE_TARGET_EMCC_WASM = 'emccwasm'

RUNTIME_X86_64_GENERIC = 'x8664generic'
RUNTIME_X86_64_NATIVE = 'x8664native'
RUNTIME_WASMTIME_CRANELIFT = 'wasmtimecranelift'
RUNTIME_WASMTIME_CRANELIFT_OPTIMIZED = 'wasmtimecraneliftoptimized'
RUNTIME_WASMTIME_LIGHTBEAM = 'wasmtimelightbeam'
RUNTIME_WASMTIME_LIGHTBEAM_OPTIMIZED = 'wasmtimelightbeamoptimized'
RUNTIME_WAVM = 'wavm'
RUNTIME_WAVM_SIMD = 'wavmsimd'
RUNTIME_WASMER = 'wasmer'
RUNTIME_WASMER_LLVM = 'wasmerllvm'
RUNTIME_WASMER_SINGLEPASS = 'wasmersinglepass'
RUNTIME_NODEJS = 'nodejs'
RUNTIME_NODEJS_WASM = 'nodejswasm'
RUNTIME_SPIDERMONKEY = 'spidermonkey'
RUNTIME_SPIDERMONKEY_WASM = 'spidermonkeywasm'
# RUNTIME_LUCET = 'lucet'
# RUNTIME_WASM3 = 'wasm3'

COMPILE_TARGETS = [COMPILE_TARGET_X86_64_GENERIC, COMPILE_TARGET_X86_64_NATIVE, COMPILE_TARGET_WASM,
                   COMPILE_TARGET_WASM_SIMD,
                   COMPILE_TARGET_JS, COMPILE_TARGET_EMCC_WASM]

COMPILE_COMMAND_TEMPLATES = {
    COMPILE_TARGET_X86_64_GENERIC: '%s -O3 -s -flto --target=x86_64-pc-linux-gnu -DPOLYBENCH_TIME -lm %%s' % CLANG_BIN_PATH,
    COMPILE_TARGET_X86_64_NATIVE: '%s -O3 -s -flto -march=native -DPOLYBENCH_TIME -lm %%s' % CLANG_BIN_PATH,
    COMPILE_TARGET_WASM: '%s -O3 -s -flto --target=wasm32-unknown-wasi --sysroot %s -DPOLYBENCH_TIME %%s' % (
        CLANG_BIN_PATH, WASI_LIBC_PATH),
    COMPILE_TARGET_WASM_SIMD: '%s -O3 -s -flto --target=wasm32-unknown-wasi --sysroot %s -DPOLYBENCH_TIME -msimd128 %%s' % (
        CLANG_BIN_PATH, WASI_LIBC_PATH),
    COMPILE_TARGET_JS: '%s -O3 -flto -DPOLYBENCH_TIME -s WASM=0 -s ALLOW_MEMORY_GROWTH=1 --memory-init-file 0 %%s' % EMCC_BIN_PATH,
    COMPILE_TARGET_EMCC_WASM: '%s -O3 -flto -DPOLYBENCH_TIME -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 %%s' % EMCC_BIN_PATH,
}

BIN_FILE_ENDING = {
    COMPILE_TARGET_X86_64_GENERIC: '',
    COMPILE_TARGET_X86_64_NATIVE: '',
    COMPILE_TARGET_WASM: '.wasm',
    COMPILE_TARGET_WASM_SIMD: '.wasm',
    COMPILE_TARGET_JS: '.js',
    COMPILE_TARGET_EMCC_WASM: '.js',
}

BENCHMARK_TYPE_BINARY_SIZE = 'binary-size'
BENCHMARK_TYPE_GZIP_BINARY_SIZE = 'gzip-binary-size'
BENCHMARK_TYPE_EXECUTION_TIME = 'execution-time'
BENCHMARK_TYPE_INITIALIZATION_TIME = 'initialization-time'

RELATIVE_BENCHMARK_C_SOURCES = [
    'datamining/covariance/covariance.c', 'datamining/correlation/correlation.c',
    'linear-algebra/kernels/mvt/mvt.c', 'linear-algebra/kernels/bicg/bicg.c',
    'linear-algebra/kernels/doitgen/doitgen.c', 'linear-algebra/kernels/2mm/2mm.c',
    'linear-algebra/kernels/atax/atax.c', 'linear-algebra/kernels/3mm/3mm.c',
    'linear-algebra/blas/gesummv/gesummv.c', 'linear-algebra/blas/syrk/syrk.c',
    'linear-algebra/blas/gemm/gemm.c', 'linear-algebra/blas/syr2k/syr2k.c',
    'linear-algebra/blas/trmm/trmm.c', 'linear-algebra/blas/gemver/gemver.c',
    'linear-algebra/blas/symm/symm.c', 'linear-algebra/solvers/trisolv/trisolv.c',
    'linear-algebra/solvers/durbin/durbin.c', 'linear-algebra/solvers/ludcmp/ludcmp.c',
    'linear-algebra/solvers/gramschmidt/gramschmidt.c',
    'linear-algebra/solvers/cholesky/cholesky.c',
    'linear-algebra/solvers/lu/lu.c', 'medley/floyd-warshall/floyd-warshall.c',
    'medley/nussinov/nussinov.c',
    'medley/deriche/deriche.c',
    'stencils/heat-3d/heat-3d.c', 'stencils/seidel-2d/seidel-2d.c',
    'stencils/jacobi-2d/jacobi-2d.c',
    'stencils/adi/adi.c', 'stencils/fdtd-2d/fdtd-2d.c', 'stencils/jacobi-1d/jacobi-1d.c'
]

BENCHMARK_C_SOURCES = list(map(lambda s: os.path.join(POLYBENCHC_PATH, s), RELATIVE_BENCHMARK_C_SOURCES))

COMPILE_TARGET_BY_RUNTIME = {
    RUNTIME_X86_64_GENERIC: COMPILE_TARGET_X86_64_GENERIC,
    RUNTIME_X86_64_NATIVE: COMPILE_TARGET_X86_64_NATIVE,
    RUNTIME_NODEJS: COMPILE_TARGET_JS,
    RUNTIME_NODEJS_WASM: COMPILE_TARGET_EMCC_WASM,
    RUNTIME_SPIDERMONKEY: COMPILE_TARGET_JS,
    RUNTIME_SPIDERMONKEY_WASM: COMPILE_TARGET_EMCC_WASM,
    RUNTIME_WASMTIME_CRANELIFT: COMPILE_TARGET_WASM,
    RUNTIME_WASMTIME_CRANELIFT_OPTIMIZED: COMPILE_TARGET_WASM,
    RUNTIME_WASMTIME_LIGHTBEAM: COMPILE_TARGET_WASM,
    RUNTIME_WASMTIME_LIGHTBEAM_OPTIMIZED: COMPILE_TARGET_WASM,
    RUNTIME_WAVM: COMPILE_TARGET_WASM,
    RUNTIME_WAVM_SIMD: COMPILE_TARGET_WASM_SIMD,
    RUNTIME_WASMER: COMPILE_TARGET_WASM,
    RUNTIME_WASMER_LLVM: COMPILE_TARGET_WASM,
    RUNTIME_WASMER_SINGLEPASS: COMPILE_TARGET_WASM,
    # RUNTIME_WASM3: COMPILE_TARGET_WASM,
    # RUNTIME_LUCET: COMPILE_TARGET_WASM,
}


def get_command(c_source: str, runtime: str, args: str = "") -> str:
    compile_target = COMPILE_TARGET_BY_RUNTIME[runtime]
    bin = get_main_binary_path_by_source(c_source, compile_target)
    if runtime == RUNTIME_X86_64_GENERIC:
        return "%s %s" % (bin, args)
    elif runtime == RUNTIME_X86_64_NATIVE:
        return "%s %s" % (bin, args)
    elif runtime == RUNTIME_NODEJS:
        return "%s %s -- %s" % (NODEJS_BIN_PATH, bin, args)
    elif runtime == RUNTIME_NODEJS_WASM:
        return "%s %s -- %s" % (NODEJS_BIN_PATH, bin, args)
    elif runtime == RUNTIME_SPIDERMONKEY:
        return "%s %s -- %s" % (SPIDERMONKEY_BIN_PATH, bin, args)
    elif runtime == RUNTIME_SPIDERMONKEY_WASM:
        return "cd tmp/polybenchc/bin/emccwasm; %s %s -- %s" % (SPIDERMONKEY_BIN_PATH, bin, args)
    elif runtime == RUNTIME_WASMTIME_CRANELIFT:
        return "%s --disable-cache %s -- %s" % (WASMTIME_BIN_PATH, bin, args)
    elif runtime == RUNTIME_WASMTIME_CRANELIFT_OPTIMIZED:
        return "%s --optimize --disable-cache %s -- %s" % (WASMTIME_BIN_PATH, bin, args)
    elif runtime == RUNTIME_WASMTIME_LIGHTBEAM:
        return "%s --lightbeam --disable-cache %s -- %s" % (WASMTIME_BIN_PATH, bin, args)
    elif runtime == RUNTIME_WASMTIME_LIGHTBEAM_OPTIMIZED:
        return "%s --lightbeam --optimize --disable-cache %s -- %s" % (WASMTIME_BIN_PATH, bin, args)
    elif runtime == RUNTIME_WAVM:
        return "%s run --nocache %s -- %s" % (WAVM_BIN_PATH, bin, args)
    elif runtime == RUNTIME_WAVM_SIMD:
        return "%s run --nocache --enable simd %s -- %s" % (WAVM_BIN_PATH, bin, args)
    elif runtime == RUNTIME_WASMER:
        return "%s run --disable-cache %s -- %s" % (WASMER_BIN_PATH, bin, args)
    elif runtime == RUNTIME_WASMER_LLVM:
        return "%s run --backend llvm --disable-cache %s -- %s" % (WASMER_BIN_PATH, bin, args)
    elif runtime == RUNTIME_WASMER_SINGLEPASS:
        return "%s run --backend singlepass --disable-cache %s -- %s" % (WASMER_BIN_PATH, bin, args)
    # elif runtime == RUNTIME_WASM3:
    #     return "%s %s -- %s" % (WASM3_BIN_PATH, bin, args)
    # elif runtime == RUNTIME_LUCET:
    #     return "docker rm -f /lucet; lucet-wasi %s -- %s" % (bin, args)
    else:
        raise Exception('unsupported compile target')


def make_output_dirs():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(MEASUREMENTS_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    os.makedirs(BIN_DIR, exist_ok=True)
    for target in COMPILE_TARGETS:
        os.makedirs(os.path.join(BIN_DIR, target), exist_ok=True)


def get_name_by_source(source_path: str) -> str:
    return os.path.splitext(os.path.basename(source_path))[0]


def get_main_binary_path_by_source(source_path: str, compile_target: str) -> str:
    name = get_name_by_source(source_path)
    return os.path.join(BIN_DIR, compile_target, name + BIN_FILE_ENDING[compile_target])


def get_all_binary_paths_by_source(source_path: str, compile_target: str) -> List[str]:
    name = get_name_by_source(source_path)
    bins = os.listdir(os.path.join(BIN_DIR, compile_target))
    bins = filter(lambda b: os.path.splitext(b)[0] == name, bins)
    bins = filter(lambda b: not b.endswith('.gz'), bins)
    bins = list(map(lambda b: os.path.join(BIN_DIR, compile_target, b), bins))
    return bins


def get_compile_command(c_source: str, compile_target: str):
    command_template = COMPILE_COMMAND_TEMPLATES[compile_target]
    source_part = "-I %s -I %s %s %s -o %s" % (
        os.path.join(POLYBENCHC_PATH, 'utilities'),
        os.path.dirname(c_source),
        os.path.join(POLYBENCHC_PATH, 'utilities', 'polybench.c'),
        c_source,
        get_main_binary_path_by_source(c_source, compile_target)
    )
    return command_template % source_part


def get_gzip_path_by_binary_path(binary_path: str):
    return binary_path + '.gz'


def benchmark_binary_size(compile_target: str):
    b = Benchmark(BENCHMARK_TYPE_BINARY_SIZE, compile_target)
    b_gzip = Benchmark(BENCHMARK_TYPE_GZIP_BINARY_SIZE, compile_target)
    for c_source in BENCHMARK_C_SOURCES:
        bins = get_all_binary_paths_by_source(c_source, compile_target)
        name = get_name_by_source(c_source)
        size = 0
        size_gzip = 0
        for bin in bins:
            size += os.path.getsize(bin)
            size_gzip += os.path.getsize(get_gzip_path_by_binary_path(bin))
        if size == 0 or size_gzip == 0:
            raise Exception('invalid file size')
        b.values[name] = size
        b_gzip.values[name] = size_gzip
        print("%s %d gzip: %d" % (name, size, size_gzip))
    b.write_csv(MEASUREMENTS_DIR, overwrite=True)
    b_gzip.write_csv(MEASUREMENTS_DIR, overwrite=True)


def compile_c_source(c_source: str, compile_target: str):
    command = get_compile_command(c_source, compile_target)
    print(command)
    os.system(command)
    bins = get_all_binary_paths_by_source(c_source, compile_target)
    for bin in bins:
        with open(bin, 'rb') as f_in, gzip.open(get_gzip_path_by_binary_path(bin), 'wb', compresslevel=9) as f_out:
            shutil.copyfileobj(f_in, f_out)


def compile_all_by_target(compile_target: str):
    for c_source in BENCHMARK_C_SOURCES:
        compile_c_source(c_source, compile_target)


def compile_all():
    for compile_target in COMPILE_TARGETS:
        compile_all_by_target(compile_target)


def benchmark_initialization_time(runtime: str):
    bi = Benchmark(BENCHMARK_TYPE_INITIALIZATION_TIME, runtime)
    for c_source in BENCHMARK_C_SOURCES:
        name = get_name_by_source(c_source)
        command = get_command(c_source, runtime, "initonly")
        print(command)
        start = perf_counter()
        exit_status = os.popen(command).close()
        end = perf_counter()
        if exit_status is not None:
            raise Exception('non zero exit status')
        initialization_time = end - start
        bi.values[name] = initialization_time
        print('initialization time: %f' % initialization_time)
    bi.write_csv(MEASUREMENTS_DIR)


def benchmark_execution_time(runtime: str):
    b = Benchmark(BENCHMARK_TYPE_EXECUTION_TIME, runtime)
    for c_source in BENCHMARK_C_SOURCES:
        name = get_name_by_source(c_source)
        command = get_command(c_source, runtime)
        print(command)
        output = os.popen(command).read()
        time = float(output.strip())
        b.values[name] = time
        print('execution time: %f' % time)
    b.write_csv(MEASUREMENTS_DIR)


def get_all_measurements() -> DataFrame:
    files = os.listdir(MEASUREMENTS_DIR)
    files = sorted(list(filter(lambda n: n.endswith('.csv'), files)))
    files = list(map(lambda b: os.path.join(MEASUREMENTS_DIR, b), files))
    data = pd.concat([pd.read_csv(f) for f in files])
    return data


def normalize_measurements_by_names(data: DataFrame, names: List[str]) -> DataFrame:
    tmp = data.copy()
    base = tmp[tmp['name'].isin(names)]
    base = base.groupby(['type', 'test']).mean().reset_index()

    def n(row: Series):
        type = row['type']
        test = row['test']
        value = row['value']
        base_value = base[(base['type'] == type) & (base['test'] == test)].iloc[0]['value']
        return value / base_value

    tmp['value'] = tmp.apply(n, axis=1)
    return tmp


def normalize_by_index(df: DataFrame, index: str, column: str) -> DataFrame:
    normal = df.loc[index][column]
    return  df.apply(lambda r: r[column] / normal, axis=1)


def aggregate_data(data: DataFrame, column: str, groupby: List[str]) -> DataFrame:
    aggregations = ['count', 'mean', 'std', 'median', 'min', 'max', 'var', 'sem']
    data = data.groupby(groupby).agg({column: aggregations})
    data.columns = aggregations
    data = data.reset_index()
    data['95error'] = data.apply(lambda r: get_confidence_error(r['count'], r['mean'], r['sem'], level=0.95), axis=1)
    return data


def aggregate_measurements(data: DataFrame) -> DataFrame:
    aggregations = ['count', 'mean', 'std', 'median', 'min', 'max', 'var', 'sem']
    data = data.groupby(['type', 'name', 'test']).agg({'value': aggregations})
    data.columns = aggregations
    data = data.reset_index()
    data['95error'] = data.apply(lambda r: get_confidence_error(r['count'], r['mean'], r['sem'], level=0.95), axis=1)
    return data


def mean_measurements(data: DataFrame) -> DataFrame:
    aggregations = ['count', 'mean', 'std', 'sem']
    data = data.groupby(['type', 'name', 'test']).agg({'value': aggregations})
    data.columns = aggregations
    data = data.reset_index()
    data['95error'] = data.apply(lambda r: get_confidence_error(r['count'], r['mean'], r['sem'], level=0.95), axis=1)
    return data

def gmean_meaned_measurements(data: DataFrame) -> DataFrame:
    data = data.groupby(['type', 'name']).agg({'mean': [stats.gmean]})
    data.columns = ['gmean']
    data = data.reset_index()
    return data
