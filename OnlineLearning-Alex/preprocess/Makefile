CC=g++
CFLAGS=-std=c++11 -Wall -g -O3

all: backgen bincoder imagereader

.PHONY: clean

clean:
	-rm *.amat
	-rm backgen bincoder

mnist.zip:
	wget http://www.iro.umontreal.ca/~lisa/icml2007data/mnist.zip

mnist_test.amat mnist_train.amat: mnist.zip
	unzip $^

backgen: backgen.cpp
	$(CC) $(CFLAGS) $^ -o $@

bincoder: bincoder.cpp
	$(CC) $(CFLAGS) $^ -o $@

imagereader: imagereader.cpp lodepng.cpp
	$(CC) $(CFLAGS) $^ -o $@
