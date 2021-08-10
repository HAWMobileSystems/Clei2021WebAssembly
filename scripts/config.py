# adjust for your system
import os

POLYBENCHC_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tmp',
                               'polybenchc', 'polybench-c-4.2.1-beta')
WASI_LIBC_PATH = '/opt/wasi-sdk-8.0/share/wasi-sysroot'
EMCC_BIN_PATH = '/home/benedikt/Development/emsdk/upstream/emscripten/emcc'
NODEJS_BIN_PATH = '/usr/bin/node'
SPIDERMONKEY_BIN_PATH = '/home/benedikt/.jsvu/spidermonkey'
WASMTIME_BIN_PATH = '/home/benedikt/.wasmtime/bin/wasmtime'
WAVM_BIN_PATH = '/usr/bin/wavm'
WASMER_BIN_PATH = '/home/benedikt/.wasmer/bin/wasmer'
CLANG_BIN_PATH = '/usr/bin/clang'
# WASM3_BIN_PATH = '/home/benedikt/wapm_packages/.bin/wasm3'
