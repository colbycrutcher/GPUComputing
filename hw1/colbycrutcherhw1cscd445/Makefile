# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -Wextra -O2

# Target executable name
TARGET = hw1

# Source files
SRCS = hw1.c functions.c

# Object files (same names, but .o instead of .c)
OBJS = hw1.o functions.o

# Default rule: build the program
all: $(TARGET)

# Link object files into final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

# Compile hw1.c into hw1.o
hw1.o: hw1.c functions.h
	$(CC) $(CFLAGS) -c hw1.c

# Compile functions.c into functions.o
functions.o: functions.c functions.h
	$(CC) $(CFLAGS) -c functions.c

# Run the program on testfile2
run: $(TARGET)
	./$(TARGET) testfile2

# Clean up compiled files
clean:
	rm -f $(TARGET) $(OBJS)
