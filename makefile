out:
	em++ lib.cpp wrapper.cpp -O3 -o module.cjs \
	-s EXPORTED_RUNTIME_METHODS='["cwrap", "ccall"]' \
	-s WASM=1 -s MODULARIZE=1

lib:
	clang++ -c lib.cpp c_wrapper.cpp -O3 -static -stdlib=libc++
	ar rcs lib.a lib.o c_wrapper.o
	rm lib.o c_wrapper.o

librust: 
	clang++ -c lib.cpp c_wrapper.cpp -O3 -stdlib=libc++ -static-libstdc++
	ar rvs liba.a lib.o c_wrapper.o
	rm lib.o c_wrapper.o
	cp liba.a T2/t2_sln_rs/lib/

t1: out
	cp module.cjs module.wasm t1
	make -C ./T1

t2: out
	cp module.cjs module.wasm t2
	make -C ./T2

test: out
	node ./test/test.js

clean:
	rm module.cjs module.wasm
	make -C ./T1 clean
	make -C ./T2 clean