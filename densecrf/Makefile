# update the path variables 

CC	= g++
CFLAGS	= -W -Wall -O2
#CFLAGS	= -W -Wall -g

all:
	make clean
	make prog_refine_pascal
	make prog_refine_pascal_v2
	make prog_refine_pascal_v4

clean:
	rm -f *.a
	rm -f *.o
	rm -f prog_test_densecrf
	rm -f prog_refine_pascal
	rm -f prog_refine_pascal_v2
	rm -f prog_refine_pascal_v4

libDenseCRF.a: libDenseCRF/bipartitedensecrf.cpp libDenseCRF/densecrf.cpp libDenseCRF/filter.cpp libDenseCRF/permutohedral.cpp libDenseCRF/util.cpp libDenseCRF/densecrf.h libDenseCRF/fastmath.h libDenseCRF/permutohedral.h libDenseCRF/sse_defs.h libDenseCRF/util.h
	$(CC) libDenseCRF/bipartitedensecrf.cpp libDenseCRF/densecrf.cpp libDenseCRF/filter.cpp libDenseCRF/permutohedral.cpp libDenseCRF/util.cpp -c $(CFLAGS)
	ar rcs libDenseCRF.a bipartitedensecrf.o densecrf.o filter.o permutohedral.o util.o

prog_test_densecrf: test_densecrf/simple_dense_inference.cpp libDenseCRF.a
	$(CC) test_densecrf/simple_dense_inference.cpp -o prog_test_densecrf $(CFLAGS) -L. -lDenseCRF

prog_refine_pascal: refine_pascal/dense_inference.cpp refine_pascal/dense_inference.h util/Timer.h libDenseCRF.a
	$(CC) refine_pascal/dense_inference.cpp -o prog_refine_pascal $(CFLAGS) -L. -lDenseCRF -I./refine_pascal/ -I./util/

prog_refine_pascal_v2: refine_pascal_v2/dense_inference.cpp util/Timer.h libDenseCRF.a
	$(CC) refine_pascal_v2/dense_inference.cpp -o prog_refine_pascal_v2 $(CFLAGS) -L. -lDenseCRF -I./util/

prog_refine_pascal_v4: refine_pascal_v4/dense_inference.cpp util/Timer.h libDenseCRF.a
	$(CC) refine_pascal_v4/dense_inference.cpp -o prog_refine_pascal_v4 $(CFLAGS) -L. -lDenseCRF -lmatio -I./util/

prog_refine_pascal_prob: refine_pascal_prob/dense_inference_prob.cpp libDenseCRF.a
	$(CC) refine_pascal_prob/dense_inference_prob.cpp -o prog_refine_pascal_prob $(CFLAGS) -L. -lDenseCRF

