
program : test.o pkeystore.o
	g++ -g -o program test.o pkeystore.o -lpthread

test.o : test.cpp pkeystore.h
	g++ -g -c test.cpp -lpthread

pkeystore.o : pkeystore.cpp pkeystore.h
	g++ -g -c pkeystore.cpp -lpthread

clean :
	rm program test.o pkeystore.o
