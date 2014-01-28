CXX = g++
CFLAGS = -g -fopenmp -O3 -fPIC -I/usr/local/include/ -I..  
LDLIBS = -g -lgomp -ljpeg 
#OBJS = JPEGPyramid.o JPEGImage.o Patchwork.o Rectangle.o PyramidStitcher.o test_stitch_pyramid.o
OBJS = JPEGPyramid.o JPEGImage.o Patchwork.o Rectangle.o PyramidStitcher.o
all:	libPyramidStitcher.so test_stitch_pyramid
#all:    libPyramidStitcher.so libPyramidStitcher.a test_stitch_pyramid

# *** test suite ***

# dynamic link option
test_stitch_pyramid: test_stitch_pyramid.o libPyramidStitcher.so
	$(CXX) -o test_stitch_pyramid test_stitch_pyramid.o -L. -lPyramidStitcher $(LDLIBS)

# static link option (doesn't like static build w/ openmp)
#test_stitch_pyramid: test_stitch_pyramid.o libPyramidStitcher.a
#	$(CXX) -o test_stitch_pyramid test_stitch_pyramid.o -lPyramidStitcher $(LDLIBS)

# OLD: link directly with .o files instead of linking with PyramidStitcher.so
#test_stitch_pyramid: $(OBJS) 
#	$(CXX) -o test_stitch_pyramid $(OBJS) $(LDLIBS)

test_stitch_pyramid.o: ../test_stitch_pyramid.cpp ../PyramidStitcher.h ../JPEGPyramid.h ../JPEGImage.h ../Patchwork.h 
	$(CXX) $(CFLAGS) -c ../test_stitch_pyramid.cpp

# *** build deployable shared object binary ***
libPyramidStitcher.so: $(OBJS) 
	$(CXX) $(OBJS) -shared -o libPyramidStitcher.so $(LDLIBS)

libPyramidStitcher.a: $(OBJS)
	$(CXX) $(OBJS) -static -o libPyramidStitcher.a $(LDLIBS)

PyramidStitcher.o: ../PyramidStitcher.cpp ../PyramidStitcher.h ../JPEGPyramid.h ../JPEGImage.h ../Patchwork.h
	$(CXX) $(CFLAGS) -c ../PyramidStitcher.cpp

JPEGPyramid.o:	../JPEGPyramid.cpp ../JPEGPyramid.h ../JPEGImage.h
	$(CXX) $(CFLAGS) -c ../JPEGPyramid.cpp 

JPEGImage.o: ../JPEGImage.cpp ../JPEGImage.h
	$(CXX) $(CFLAGS) -c ../JPEGImage.cpp 

Patchwork.o: ../Patchwork.cpp ../Patchwork.h ../JPEGPyramid.h ../Rectangle.h
	$(CXX) $(CFLAGS) -c ../Patchwork.cpp 

Rectangle.o: ../Rectangle.cpp ../Rectangle.h
	$(CXX) $(CFLAGS) -c ../Rectangle.cpp 

clean:
	rm -f *~ *.so *.o *.mex* *.obj 

