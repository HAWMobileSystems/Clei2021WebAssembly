from polybenchc_common import make_output_dirs, benchmark_initialization_time, RUNTIME_X86_64_NATIVE, \
    RUNTIME_X86_64_GENERIC, RUNTIME_NODEJS, RUNTIME_SPIDERMONKEY, \
    RUNTIME_WASMTIME_CRANELIFT_OPTIMIZED, RUNTIME_WASMTIME_CRANELIFT, RUNTIME_WAVM, RUNTIME_WAVM_SIMD, RUNTIME_WASMER, \
    RUNTIME_WASMER_LLVM, benchmark_binary_size, benchmark_execution_time, COMPILE_TARGET_X86_64_GENERIC, \
    COMPILE_TARGET_X86_64_NATIVE, COMPILE_TARGET_WASM, COMPILE_TARGET_WASM_SIMD, COMPILE_TARGET_JS, \
    COMPILE_TARGET_EMCC_WASM, compile_all, RUNTIME_WASMER_SINGLEPASS, compile_all_by_target, RUNTIME_NODEJS_WASM, \
    RUNTIME_SPIDERMONKEY_WASM

make_output_dirs()

# compile_all_by_target(COMPILE_TARGET_X86_64_GENERIC)
# compile_all_by_target(COMPILE_TARGET_WASM)
# compile_all_by_target(COMPILE_TARGET_WASM_SIMD)
# compile_all_by_target(COMPILE_TARGET_JS)
# compile_all_by_target(COMPILE_TARGET_EMCC_WASM)

# benchmark_binary_size(COMPILE_TARGET_X86_64_GENERIC)
# benchmark_binary_size(COMPILE_TARGET_WASM)
# benchmark_binary_size(COMPILE_TARGET_WASM_SIMD)
# benchmark_binary_size(COMPILE_TARGET_JS)
# benchmark_binary_size(COMPILE_TARGET_EMCC_WASM)

# benchmark_initialization_time(RUNTIME_X86_64_GENERIC)
# benchmark_initialization_time(RUNTIME_NODEJS)
# benchmark_initialization_time(RUNTIME_NODEJS_WASM)
# benchmark_initialization_time(RUNTIME_SPIDERMONKEY)
# benchmark_initialization_time(RUNTIME_SPIDERMONKEY_WASM)
# benchmark_initialization_time(RUNTIME_WASMTIME_CRANELIFT)
# benchmark_initialization_time(RUNTIME_WASMTIME_CRANELIFT_OPTIMIZED)
# benchmark_initialization_time(RUNTIME_WAVM)
# benchmark_initialization_time(RUNTIME_WAVM_SIMD)
# benchmark_initialization_time(RUNTIME_WASMER)
# benchmark_initialization_time(RUNTIME_WASMER_LLVM)
# benchmark_initialization_time(RUNTIME_WASMER_SINGLEPASS)

# benchmark_execution_time(RUNTIME_X86_64_GENERIC)
# benchmark_execution_time(RUNTIME_NODEJS)
# benchmark_execution_time(RUNTIME_NODEJS_WASM)
# benchmark_execution_time(RUNTIME_SPIDERMONKEY)
# benchmark_execution_time(RUNTIME_SPIDERMONKEY_WASM)
# benchmark_execution_time(RUNTIME_WASMTIME_CRANELIFT)
# benchmark_execution_time(RUNTIME_WASMTIME_CRANELIFT_OPTIMIZED)
# benchmark_execution_time(RUNTIME_WAVM)
# benchmark_execution_time(RUNTIME_WAVM_SIMD)
# benchmark_execution_time(RUNTIME_WASMER)
# benchmark_execution_time(RUNTIME_WASMER_LLVM)
# benchmark_execution_time(RUNTIME_WASMER_SINGLEPASS)
