NVCC ?= nvcc
ARCH ?= sm_89

NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -arch=$(ARCH)
LDFLAGS ?= -lcublas

SRCS := $(wildcard sgemm_*.cu)
BINS := $(SRCS:.cu=)

.PHONY: all clean list

all: $(BINS)

# sgemm_13 uses a 512-thread block and needs <=128 regs/thread, otherwise
# the launch fails on RTX 4070 due to register file limits.
sgemm_13_db_b256n128_bk16: NVCCFLAGS += --maxrregcount=128
sgemm_14_sb_b256n128_bk16_pad8: NVCCFLAGS += --maxrregcount=128
sgemm_15_sb_b256n128_bk32: NVCCFLAGS += --maxrregcount=128
sgemm_16_sb_b256n128_bk32_vec4a: NVCCFLAGS += --maxrregcount=128
sgemm_17_db_b256n128_bk32_optin: NVCCFLAGS += --maxrregcount=128
sgemm_18_db_b256n128_bk32_optin_staged_prefetch: NVCCFLAGS += --maxrregcount=128
sgemm_19_db_b256n128_tm16tn8_rc128: NVCCFLAGS += --maxrregcount=128
sgemm_20_db_b256n128_bk16_pad8_optin: NVCCFLAGS += --maxrregcount=128
sgemm_24_db_b256n128_bk16_pad8_optin_cp_async_b: NVCCFLAGS += --maxrregcount=128
sgemm_22_db_b256n128_bk16_pad8_warp_broadcast: NVCCFLAGS += --maxrregcount=128
sgemm_25_dispatch_autotune_zoo: NVCCFLAGS += --maxrregcount=128

list:
	@printf "%s\n" $(BINS)

%: %.cu sgemm_bench.cuh sgemm_main.cuh
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(BINS)
