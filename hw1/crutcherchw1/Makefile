#Colby Crutcher

hw1: hw1.o functions.o
	gcc -o hw1 hw1.o functions.o

hw1.o: hw1.c functions.h
	gcc -c hw1.c

functions.o: functions.c functions.h
	gcc -c functions.c

clean:
	rm -f hw1 hw1.o functions.o
