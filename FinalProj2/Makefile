NVCC        = nvcc
CXX         = g++
NVCCFLAGS   = -arch=sm_75 -std=c++17 -O2
CXXFLAGS    = -std=c++14 -O2

TARGET      = mergesort

CUDA_SRCS   = main.cu kernel.cu
CPP_SRCS    = cpumergesort.cpp

CUDA_OBJS   = $(CUDA_SRCS:.cu=.o)
CPP_OBJS    = $(CPP_SRCS:.cpp=.o)

ALL_OBJS    = $(CUDA_OBJS) $(CPP_OBJS)

all: $(TARGET)

$(TARGET): $(ALL_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(ALL_OBJS) $(TARGET)

.PHONY: all clean
