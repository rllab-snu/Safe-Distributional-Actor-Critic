g++ -fPIC -c main.cpp
g++ -shared -o main.so main.o
rm *.o