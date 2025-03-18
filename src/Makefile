###
# Damien ASSIRE  - 21112838
# Marie BONBOIRE - 21100552
###

# Sources
# --------------------------------------------------------------------------------------------------

OBJ_SRC = scalar_vector_32.o dot_product_32.o scalar_vector_mod_32.o butterfly_fft.o
OBJ_PROF = profiler.o
OBJ_TEST = tests/test_dot_product.o tests/test_dp.o tests/test_sc_mod.o tests/test_split.o tests/test_fft.o

PROG_PROF = profiler
PROG_TEST = tests/test_dot_product tests/test_dp tests/test_sc_mod tests/test_split tests/test_fft

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
#CFLAGS += -Wpedantic
#CFLAGS += -Wconversion -Wsign-conversion
#CFLAGS += -Wcast-qual -Wcast-align
#CFLAGS += -Wundef -Wunused
#CFLAGS += -Wdouble-promotion
#CFLAGS += -fopt-info

LFLAGS  = -fwhole-program#
LFLAGS += -L/$(HOME)/lib/ -lflint -lgmp -lmpfr -lpthread

# Rules 
# --------------------------------------------------------------------------------------------------
all: $(ALL_OBJ) $(ALL_PROG)

tests: $(OBJ_SRC) $(OBJ_TEST) $(PROG_TEST)

profile: $(OBJ_SRC) $(OBJ_PROF) $(PROG_PROF)

clean:
	-rm -r *~ Makefile.bak $(ALL_PROG) *.i *.o tests/*.o 2> /dev/null || true

# Rules for the mains
# --------------------------------------------------------------------------------------------------
tests/test_dp: tests/test_dp.o dot_product_32.o
	$(CC) -o $@ $^ $(LFLAGS)

tests/test_dot_product: tests/test_dot_product.o dot_product_32.o
	$(CC) -o $@ $^ $(LFLAGS)

tests/test_sc_mod: tests/test_sc_mod.o scalar_vector_mod_32.o
	$(CC) -o $@ $^ $(LFLAGS)

tests/test_split: tests/test_split.o
	$(CC) -o $@ $^ $(LFLAGS)

tests/test_fft: tests/test_fft.o butterfly_fft.o
	$(CC) -o $@ $^ $(LFLAGS)

profiler: profiler.o scalar_vector_32.o dot_product_32.o scalar_vector_mod_32.o
	$(CC) -o $@ $^ $(LFLAGS)

profile_dot_unrolled: profile_dot_unrolled.o dot_product_32_unrolled.o
	$(CC) -o $@ $^ $(LFLAGS)
