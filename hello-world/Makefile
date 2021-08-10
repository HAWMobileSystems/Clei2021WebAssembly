default: bin/hello-rust bin/hello-rust.wasm bin/hello-c bin/hello-c.wasm

bin:
	mkdir bin

bin/hello-rust: src/hello-rust.rs bin
	rustc -C debuginfo=0 -C opt-level=3 -C lto src/hello-rust.rs -o bin/hello-rust
	strip -s bin/hello-rust

bin/hello-rust.wasm: src/hello-rust.rs bin
	rustc -C debuginfo=0 -C opt-level=3 --target=wasm32-wasi -C lto src/hello-rust.rs -o bin/hello-rust.wasm
	wasm-strip bin/hello-rust.wasm
	#wasm-opt -O3 -o bin/hello-rust.wasm bin/hello-rust.wasm

bin/hello-c: src/hello-c.c bin
	clang -s -O3 -flto src/hello-c.c -o bin/hello-c
	strip -s bin/hello-c

# lto fails
bin/hello-c.wasm: src/hello-c.c bin
	clang -s -O3 --target=wasm32-unknown-wasi --sysroot /opt/wasi-libc src/hello-c.c -o bin/hello-c.wasm
	wasm-strip bin/hello-c.wasm
	wasm2wat bin/hello-c.wasm > bin/hello-c.wat
	#wasm-opt -O3 -o bin/hello-c.wasm bin/hello-c.wasm

clear:
	rm -rf bin
