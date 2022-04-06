CFLAGS = -g -Wall -std=c++11 -pthread
OFLAGS = -O2

all: main

main: main.cpp data.cpp estimator.cpp
	g++ $(CFLAGS) $(OFLAGS) -o main main.cpp data.cpp estimator.cpp

clean:
	rm -f main