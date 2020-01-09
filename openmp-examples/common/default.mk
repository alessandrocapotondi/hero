############################ OpenMP Sources ############################
ifeq ($(compiler),gcc)
  CC := riscv32-hero-unknown-elf-gcc
else
  CC := clang
endif
LINK := llvm-link
COB := clang-offload-bundler
DIS := llvm-dis

TARGET_HOST = ${HERO_TOOLCHAIN_HOST_TARGET}
TARGET_DEV = riscv32-hero-unknown-elf

ARCH_HOST = host-$(TARGET_HOST)
ARCH_DEV = openmp-$(TARGET_DEV)

HOST_OBJDUMP := $(TARGET_HOST)-objdump
DEV_OBJDUMP := $(TARGET_DEV)-objdump

ifeq ($(strip $(default-as)),)
ifeq ($(only),pulp)
  default-as=pulp
else
  default-as=host
endif
endif

ifeq ($(strip $(opt)),)
  opt = 3
endif

.DEFAULT_GOAL = all
DEFMK_ROOT := $(patsubst %/,%, $(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# CFLAGS and LDFLAGS have three components/stages
# 1) without suffix, they apply to heterogeneous compilation;
# 3) with _PULP suffix, they apply only to the PULP part of compilation;
# 4) with _COMMON suffix, they apply to both PULP and host compilation.
CFLAGS_COMMON += $(cflags) -O$(opt)
LDFLAGS_COMMON ?= $(ldflags)
ifeq ($(default-as),pulp)
	ifeq ($(compiler),gcc)
		CFLAGS_COMMON += -D__device= -D__PULP__
	else
		CFLAGS_COMMON += -fhero-device-default-as=device
	endif
endif
ifeq ($(compiler),gcc)
	CFLAGS_COMMON += -fopenmp
	CFLAGS_PULP += $(CFLAGS_COMMON) -march=rv32imafcXpulpv2
	CFLAGS += $(CFLAGS_COMMON) -fopenmp-targets=$(TARGET_DEV)
	LDFLAGS_PULP += $(LDFLAGS_COMMON) -lgomp
else
	CFLAGS_COMMON += -fopenmp=libomp
	CFLAGS_PULP += $(CFLAGS_COMMON) -target $(TARGET_DEV)
	CFLAGS += -target $(TARGET_HOST) $(CFLAGS_COMMON) -fopenmp-targets=$(TARGET_DEV)
	LDFLAGS_PULP += $(LDFLAGS_COMMON)
endif
LDFLAGS += $(LDFLAGS_COMMON) -lhero-target
ifeq ($(TARGET_HOST),riscv64-hero-linux-gnu)
  # FIXME: we explicitly need to embed the correct linker for riscv
  LDFLAGS += -Wl,-dynamic-linker,/lib/ld-linux-riscv64-lp64.so.1
endif

INCPATHS += -I$(DEFMK_ROOT) -include hero_64.h
ifeq ($(compiler),gcc)
	INCPATHS += -I$(DEFMK_ROOT)/../../support/libhero-target/inc
	INCPATHS += -I$(DEFMK_ROOT)/../common/gcc
endif
LIBPATHS ?=

BENCHMARK = $(shell basename `pwd`)
EXE = $(BENCHMARK)
SRC = $(CSRCS)

DEPDIR := .deps
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d
# DEPFLAGS = -MMD -MP -MF $(DEPDIR)/$*.d

AS_ANNOTATE_ARGS ?=

only ?= # can be set to `pulp` to compile a binary only for PULP

.PHONY: all exe clean
.PRECIOUS: %.ll %.o

ifeq ($(only),pulp)
all : $(DEPS) $(EXE) $(EXE).dis slm

%.o: %.c $(DEPDIR)/%.d | $(DEPDIR)
	$(CC) -c $(DEPFLAGS) $(CFLAGS_PULP) $(INCPATHS) -mnativeomp $<

%.ll: %.c $(DEPDIR)/%.d | $(DEPDIR)
	$(CC) -c -emit-llvm -S $(DEPFLAGS) $(CFLAGS_PULP) $(INCPATHS) $<

%.OMP.ll: %.ll
	hc-omp-pass $< OmpAddressSpaceAssigner "HERCULES-omp-address-space-assigner" $(<:.ll=.TMP.1.ll) $(AS_ANNOTATE_ARGS)
	hc-omp-pass $(<:.ll=.TMP.1.ll) OmpKernelWrapper "HERCULES-omp-kernel-wrapper" $(<:.ll=.TMP.2.ll)
	hc-omp-pass $(<:.ll=.TMP.2.ll) OmpHostPointerLegalizer "HERCULES-omp-host-pointer-legalizer" $(<:.ll=.TMP.3.ll)
	cp $(<:.ll=.TMP.3.ll) $(<:.ll=.OMP.ll)

ifeq ($(compiler),gcc)
$(EXE): $(SRC:.c=.o)
# 	touch libgomp.spec
	clang $(LIBPATHS) $(CFLAGS_PULP) -target $(TARGET_DEV) $^ $(LDFLAGS_PULP) -o $@
else
$(EXE): $(SRC:.c=.OMP.ll)
	$(CC) $(LIBPATHS) $(CFLAGS_PULP) $^ $(LDFLAGS_PULP) -o $@
endif

slm: $(EXE)_l1.slm $(EXE)_l2.slm

$(EXE)_l2.slm: $(EXE)
	$(DEV_OBJDUMP) -s --start-address=0x1c000000 --stop-address=0x1cffffff $^ | rg '^ ' | cut -c 2-45 \
      | sort \
      > $@
	$(DEFMK_ROOT)/one_word_per_line.py $@

$(EXE)_l1.slm: $(EXE)
	$(DEV_OBJDUMP) -s --start-address=0x10000000 --stop-address=0x1bffffff $^ | rg '^ ' | cut -c 2-45 \
      | perl -p -e 's/^1b/10/' \
      | sort \
      > $@
	$(DEFMK_ROOT)/one_word_per_line.py $@

$(EXE).dis: $(EXE)
	$(DEV_OBJDUMP) -d $^ > $@

else
all: $(DEPS) $(EXE) $(EXE).dis

%.ll: %.c $(DEPDIR)/%.d | $(DEPDIR)
	$(CC) -c -emit-llvm -S $(DEPFLAGS) $(CFLAGS) $(INCPATHS) $<
	$(COB) -inputs=$@ -outputs="$(<:.c=-host.ll),$(<:.c=-dev.ll)" -type=ll -targets="$(ARCH_HOST),$(ARCH_DEV)" -unbundle

%-dev.OMP.ll: %.ll
	hc-omp-pass $(<:.ll=-dev.ll) OmpAddressSpaceAssigner "HERCULES-omp-address-space-assigner" $(@:.OMP.ll=.TMP.1.ll)
	hc-omp-pass $(@:.OMP.ll=.TMP.1.ll) OmpKernelWrapper "HERCULES-omp-kernel-wrapper" $(@:.OMP.ll=.TMP.2.ll)
	hc-omp-pass $(@:.OMP.ll=.TMP.2.ll) OmpHostPointerLegalizer "HERCULES-omp-host-pointer-legalizer" $(@:.OMP.ll=.TMP.3.ll)
	cp $(@:.OMP.ll=.TMP.3.ll) $@

%-host.OMP.ll: %.ll
	hc-omp-pass $(<:.ll=-host.ll) OmpKernelWrapper "HERCULES-omp-kernel-wrapper" $(@:.OMP.ll=.TMP.1.ll)
	cp $(@:.OMP.ll=.TMP.1.ll) $@

%-out.ll: %-host.OMP.ll %-dev.OMP.ll
	$(COB) -inputs="$(@:-out.ll=-host.OMP.ll),$(@:-out.ll=-dev.OMP.ll)" -outputs=$@ -type=ll -targets="$(ARCH_HOST),$(ARCH_DEV)"

$(EXE): $(SRC:.c=-out.ll)
	$(CC) $(LIBPATHS) $(CFLAGS) $< $(LDFLAGS) -o $@

$(EXE).dis: $(EXE)
	$(HOST_OBJDUMP) -d $^ > $@

endif

$(DEPDIR):
	@mkdir -p $@

DEPFILES := $(CSRCS:%.c=$(DEPDIR)/%.d)
$(DEPFILES):

include $(wildcard $(DEPFILES))

clean::
	-rm -vf __hmpp* $(EXE) *~ *.dis *.ll *.slm *.o *.s
	-rm -rvf $(DEPDIR)

init-target-host:
ifndef HERO_TARGET_HOST
	$(error HERO_TARGET_HOST is not set)
endif
	ssh -t $(HERO_TARGET_HOST) './sourceme.sh'
	ssh -t $(HERO_TARGET_HOST) 'rmmod -f pulp'
	ssh -t $(HERO_TARGET_HOST) 'insmod $(HERO_TARGET_PATH_DRIVER)/pulp.ko'

prepare:: init-target-host $(EXE)
ifndef HERO_TARGET_HOST
	$(error HERO_TARGET_HOST is not set)
endif
	ssh $(HERO_TARGET_HOST) "mkdir -p $(HERO_TARGET_PATH_APPS)"
	scp $(EXE) $(HERO_TARGET_HOST):$(HERO_TARGET_PATH_APPS)/.

run:: prepare $(EXE)
ifeq ($(call ifndef_any_of,HERO_TARGET_HOST HERO_TARGET_PATH_APPS HERO_TARGET_PATH_LIB),)
	ssh -t $(HERO_TARGET_HOST) 'export LD_LIBRARY_PATH='"'$(HERO_TARGET_PATH_LIB)'"'; cd ${HERO_TARGET_PATH_APPS}; ./$(EXE) $(RUN_ARGS)'
else
	$(error HERO_TARGET_HOST and/or HERO_TARGET_PATH_APPS is not set)
endif
