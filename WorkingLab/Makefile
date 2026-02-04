NVCC = nvcc
NVCCFLAGS = -arch=sm_86

TARGET = myPaint

SRCS = main.cu pgmUtility.cu pgmProcess.cu
OBJS = main.o pgmUtility.o pgmProcess.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(OBJS)

main.o: main.cu pgmUtility.h pgmProcess.h
	$(NVCC) $(NVCCFLAGS) -c main.cu

pgmUtility.o: pgmUtility.cu pgmUtility.h pgmProcess.h
	$(NVCC) $(NVCCFLAGS) -c pgmUtility.cu

pgmProcess.o: pgmProcess.cu pgmProcess.h
	$(NVCC) $(NVCCFLAGS) -c pgmProcess.cu

clean:
	rm -f *.o $(TARGET)
