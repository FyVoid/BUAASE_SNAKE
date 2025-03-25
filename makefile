out:
	emcc lib.cpp wrapper.cpp -O3 -o module.cjs \
	-s EXPORTED_RUNTIME_METHODS='["cwrap", "ccall"]' \
	-s WASM=1 -s MODULARIZE=1

t1: out
	cp module.cjs module.wasm bridge.js t1
	make -C ./T1

test: out
	node ./test/test.js

clean:
	rm module.cjs module.wasm
	make -C ./T1 clean