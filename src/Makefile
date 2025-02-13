###
# Damien ASSIRE  - 21112838
# Marie BONBOIRE - 21100552
###

# Sources
# --------------------------------------------------------------------------------------------------

OBJ = scalar_vector_32.o test_flint.o profile_cl.o profile_sec.o profile_sec2.o
PROG = test_flint profile_cl profile_sec profile_sec2
#FUN = naive_load.o simd_add.o seq_vector_vector.o

# Tools and parameters definitions
# --------------------------------------------------------------------------------------------------
CC = gcc

CFLAGS  = -c#                       # Stop after compilation, then produce .o
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

#DFLAGS  = -g#                       # Useful for gdb
#DFLAGS += -O3#                      # No optimisation mode for compiler
#
#RFLAGS  = -O3#                      # Full optimisation mode of compiler

# Rules (here they are used such as simple shell scripts)
# --------------------------------------------------------------------------------------------------

compil: mkdir_obj mkdir_debug all

mkdir_obj:
	@test -d obj || mkdir obj

mkdir_debug:
	@test -d debug || mkdir debug
	@test -d debug/obj || mkdir debug/obj

clean:
	-rm -r obj debug *~ Makefile.bak $(PROG) *.i *.o 2> /dev/null || true

# Rules with automatic variables
# --------------------------------------------------------------------------------------------------

all : $(OBJ) $(PROG)

%.o : %.c
	$(CC) -o $@ $(CFLAGS) $^

obj/%.o : %.c
	$(CC) -o $@ $(CFLAGS) $(RFLAGS) -save-temps -fverbose-asm $^

asm: mkdir_obj $(addprefix obj/,$(OBJ) $(FUN))

# Rules for the mains
# --------------------------------------------------------------------------------------------------
test_flint: test_flint.o scalar_vector_32.o
	$(CC) -o $@ $^ $(LFLAGS)

profile_cl: profile_cl.o scalar_vector_32.o
	$(CC) -o $@ $^ $(LFLAGS)

profile_sec: profile_sec.o scalar_vector_32.o
	$(CC) -o $@ $^ $(LFLAGS)

profile_sec2: profile_sec2.o scalar_vector_32.o
	$(CC) -o $@ $^ $(LFLAGS)
