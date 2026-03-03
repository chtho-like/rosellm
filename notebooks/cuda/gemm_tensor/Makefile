NVCC ?= nvcc
ARCH ?= sm_89

NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -arch=$(ARCH)
LDFLAGS ?= -lcublas

SRCS := $(wildcard hgemm_*.cu)
BINS := $(SRCS:.cu=)

.PHONY: all clean list

all: $(BINS)

list:
	@printf "%s\n" $(BINS)

%: %.cu hgemm_bench.cuh hgemm_main.cuh
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(BINS)
