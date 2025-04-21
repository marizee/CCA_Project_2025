###
# Damien ASSIRE  - 21112838
# Marie BONBOIRE - 21100552
###

# Sources
# --------------------------------------------------------------------------------------------------

#OBJ_SRC = scalar_vector_32.o dot_product_32.o scalar_vector_mod_32.o butterfly_fft_32.o butterfly_fft_64.o lazy_butterfly_fft_64.o mulsplit.o
OBJ_SRC = add_64.o add_mod_64.o
OBJ_PROF = profiler.o
#OBJ_TEST = tests/test_dot_product.o tests/test_dp.o tests/test_sc_mod.o tests/test_split.o tests/test_fft_32.o tests/test_fft_64.o tests/test_lazy_fft_64.o tests/test_dot_product_mod.o tests/test_mulsplit.o
OBJ_TEST = tests/test_add.o tests/test_add_mod.o

PROG_PROF = profiler
#PROG_TEST = tests/test_dot_product tests/test_dp tests/test_sc_mod tests/test_split tests/test_fft_32 tests/test_fft_64 tests/test_lazy_fft_64 tests/test_dot_product_mod tests/test_mulsplit
PROG_TEST = tests/test_add tests/test_add_mod

ALL_OBJ = $(OBJ_SRC) $(OBJ_PROF) $(OBJ_TEST)
ALL_PROG = $(PROG_PROF) $(PROG_TEST)

# Tools and parameters definitions
# --------------------------------------------------------------------------------------------------
CC = gcc

CFLAGS = -c#                       # Stop after compilation, then produce .o
CFLAGS += -Wall -Wextra #-Werror#    # Gives almost all C warnings and considers them to be errors
CFLAGS += -O3
CFLAGS += -march=native#            # Enable AVX 
CFLAGS += -Wno-unused-parameter# 
CFLAGS += -I/$(HOME)/include/
#CFLAGS += -I/dsk/l1/vneiger/include/
#CFLAGS += -Wpedantic
#CFLAGS += -Wconversion -Wsign-conversion
#CFLAGS += -Wcast-qual -Wcast-align
#CFLAGS += -Wundef -Wunused
#CFLAGS += -Wdouble-promotion
#CFLAGS += -fopt-info

LFLAGS  = -fwhole-program#
LFLAGS += -L/$(HOME)/lib/ -lflint -lgmp -lmpfr -lpthread
#LFLAGS += -L/dsk/l1/vneiger/lib/ -lflint -lgmp -lmpfr -lpthread

# Rules 
# --------------------------------------------------------------------------------------------------
all: $(ALL_OBJ) $(ALL_PROG)

tests: $(OBJ_SRC) $(OBJ_TEST) $(PROG_TEST)

profile: $(OBJ_SRC) $(OBJ_PROF) $(PROG_PROF)

clean:
	-rm -r *~ Makefile.bak $(ALL_PROG) *.i *.o tests/*.o 2> /dev/null || true

# Rules for the mains
# --------------------------------------------------------------------------------------------------
tests/test_add: tests/test_add.o add_64.o
	$(CC) -o $@ $^ $(LFLAGS)

tests/test_add_mod: tests/test_add_mod.o add_mod_64.o
	$(CC) -o $@ $^ $(LFLAGS)

tests/test_dp: tests/test_dp.o dot_product_32.o
	$(CC) -o $@ $^ $(LFLAGS)

tests/test_dot_product: tests/test_dot_product.o dot_product_32.o
	$(CC) -o $@ $^ $(LFLAGS)

tests/test_sc_mod: tests/test_sc_mod.o scalar_vector_mod_32.o
	$(CC) -o $@ $^ $(LFLAGS)

tests/test_split: tests/test_split.o
	$(CC) -o $@ $^ $(LFLAGS)

tests/test_fft_32: tests/test_fft_32.o butterfly_fft_32.o
	$(CC) -o $@ $^ $(LFLAGS)

tests/test_fft_64: tests/test_fft_64.o butterfly_fft_64.o
	$(CC) -o $@ $^ $(LFLAGS)

tests/test_lazy_fft_64: tests/test_lazy_fft_64.o lazy_butterfly_fft_64.o
	$(CC) -o $@ $^ $(LFLAGS)

tests/test_mulsplit: tests/test_mulsplit.o mulsplit.o
	$(CC) -o $@ $^ $(LFLAGS)

tests/test_dot_product_mod: tests/test_dot_product_mod.o dot_product_32_mod.o
	$(CC) -o $@ $^ $(LFLAGS)

profiler: profiler.o scalar_vector_32.o dot_product_32.o scalar_vector_mod_32.o butterfly_fft_64.o lazy_butterfly_fft_64.o dot_product_32_mod.o
	$(CC) -o $@ $^ $(LFLAGS)
