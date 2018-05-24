CC=clang
LD=g++
CFLAGS=-O3 -c -msse2 -msse -msse4.1 -msse3 -DUSE_SSE2 `pkg-config --cflags opencv` -I /usr/local/Cellar/opencv/3.4.1_4/include/opencv2 
CFLAGS1=-O3 -c -msse2 -msse -msse4.1 -msse3 `pkg-config --cflags opencv` -I /usr/local/Cellar/opencv/3.4.1_4/include/opencv2 
LDFLAGS=`pkg-config --libs opencv`

all:io.o canny.o main.o program
BIN:program
io.o:io.cpp io.hpp
	$(CC) $(CFLAGS) $^ 
canny.o:canny.cpp canny.hpp
	$(CC) $(CFLAGS) $^
main.o:main.cpp io.hpp
	$(CC) $(CFLAGS1) $^
program:main.o io.o canny.o
	$(LD) $(LDFLAGS) $^ -o program

clean:
	##rm program
	find . -name "*.o" -type f -delete
	find . -name "*.gch" -type f -delete #need to refresh if you change the hpp files
cleanphoto:
	find . -name "*.jpg" -type f -delete
