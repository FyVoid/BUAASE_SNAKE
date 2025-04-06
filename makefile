out:
	em++ -DT1 -DT2 -DT3 lib.cpp wrapper.cpp -O3 -o module.cjs \
	-s ALLOW_MEMORY_GROWTH=1 \
	-s TOTAL_MEMORY=1Gb \
	-s STACK_SIZE=52428800 \
	-s SAFE_HEAP=1 -g \
	-s EXPORTED_RUNTIME_METHODS='["cwrap", "ccall"]' \
	-s WASM=1 \
	-s MODULARIZE=1

lib_t1:
	em++ -DT1 lib.cpp wrapper.cpp -O3 -o module_t1.cjs \
	-s ALLOW_MEMORY_GROWTH=1 \
	-s TOTAL_MEMORY=1Gb \
	-s STACK_SIZE=52428800 \
	-s SAFE_HEAP=1 -g \
	-s EXPORTED_RUNTIME_METHODS='["cwrap", "ccall"]' \
	-s WASM=1 \
	-s MODULARIZE=1

lib_t2:
	em++ -DT2 lib.cpp wrapper.cpp -O3 -o module_t2.cjs \
	-s ALLOW_MEMORY_GROWTH=1 \
	-s TOTAL_MEMORY=1Gb \
	-s STACK_SIZE=52428800 \
	-s SAFE_HEAP=1 -g \
	-s EXPORTED_RUNTIME_METHODS='["cwrap", "ccall"]' \
	-s WASM=1 \
	-s MODULARIZE=1

lib_t3:
	em++ -DT3 lib.cpp wrapper.cpp -O3 -o module_t3.cjs \
	-s ALLOW_MEMORY_GROWTH=1 \
	-s TOTAL_MEMORY=1Gb \
	-s STACK_SIZE=52428800 \
	-s SAFE_HEAP=1 -g \
	-s EXPORTED_RUNTIME_METHODS='["cwrap", "ccall"]' \
	-s WASM=1 \
	-s MODULARIZE=1

t1: lib_t1
	cp module_t1.cjs module_t1.wasm t1
	make -C ./T1

t2: lib_t2
	cp module_t2.cjs module_t2.wasm t2
	make -C ./T2

t3: lib_t3
	cp module_t3.cjs module_t3.wasm t3
	make -C ./T3

t3-gui: lib_t3
	cp module_t3.cjs module_t3.wasm t3
	make -C ./T3 test-gui

test: lib_t1 lib_t2 lib_t3
	cp module_t1.cjs module_t1.wasm t1
	cp module_t2.cjs module_t2.wasm t2
	cp module_t3.cjs module_t3.wasm t3
	make -C ./T1 test
	make -C ./T2 test
	make -C ./T3 test

clean:
	rm *.cjs *.wasm
	make -C ./T1 clean
	make -C ./T2 clean
	make -C ./T3 clean