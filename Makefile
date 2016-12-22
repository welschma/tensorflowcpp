CXX=g++

CXXFLAGS+= -g -std=c++11
CXXFLAGS+= -Wall -pedantic
CXXFLAGS+= -I /usr/local/include/google/tensorflow -I /usr/local/include/eigen/eigen-eigen-9569d1f35bae
#CXXFLAGS+= -Wl,--verbose
CXXFLAGS+= -L/usr/local/lib/

BIN=use_graph
LIBS=-ltensorflow 

SRC=$(wildcard *.cpp)
OBJ=$(SRC:%.cpp=%.o)

all: $(OBJ)
	$(CXX) $^ -o $(BIN) $(CXXFLAGS) $(LIBS)  

# testneu: test0.o 
# 	g++  $^ -o $@ -Wl,--verbose -L/usr/local/lib/ -ltensorflow 

%.o: %.c
	$(CXX) $@ -c $<

clean:
	rm -f *.o
	rm $(BIN)
