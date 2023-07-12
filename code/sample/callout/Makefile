# Makefile for GCC
#
# Author: Guilherme Domingos Faria Silva
#
CC=gcc
CFLAGS=-std=c++11
INCLUDEPATH=../../include
LIBPATH=-L../../lib
INCLUDE=-I$(INCLUDEPATH)
LIBS=-lstdc++ -larboretum -lm
SRC= app.cpp callout.cpp object.cpp
OBJS=$(subst .cpp,.o,$(SRC))


# Implicit Rules
%.o: %.cpp $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@ $(INCLUDE)

callout: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o callout $(INCLUDE) $(LIBPATH) $(LIBS)

clean:
	rm -f *.o
	rm -f callout
