NVCC ?= nvcc
ARCH ?= sm_89

NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -arch=$(ARCH) --use_fast_math

SRCS := $(wildcard flashattn_*.cu)
BINS := $(SRCS:.cu=)

.PHONY: all clean list

all: $(BINS)

list:
	@printf "%s\n" $(BINS)

%: %.cu flashattn_bench.cuh flashattn_main.cuh
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(BINS)

