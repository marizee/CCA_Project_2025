###
# Damien ASSIRE  - 21112838
# Marie BONBOIRE - 21100552
###

# Sources
# --------------------------------------------------------------------------------------------------

OBJ = scalar_vector_32.o vector_vector_32.o gather.o
PROG = scalar_vector_32 vector_vector_32 gather
FUN = naive_load.o simd_add.o seq_vector_vector.o

# Tools and parameters definitions
# --------------------------------------------------------------------------------------------------

CC = gcc

CFLAGS  = -c#                       # Stop after compilation, then produce .o
CFLAGS += -Wall -Wextra -Werror#    # Gives almost all C warnings and considers them to be errors
CFLAGS += -g#                       # Useful for gdb
CFLAGS += -O3#                      # No optimisation mode for compiler
CFLAGS += -march=native#            # Enable AVX 
CFLAGS += -Wno-unused-parameter# 
CFLAGS += -Wpedantic
CFLAGS += -Wconversion -Wsign-conversion
CFLAGS += -Wcast-qual -Wcast-align
CFLAGS += -Wundef -Wunused
CFLAGS += -Wdouble-promotion

LFLAGS  = -fwhole-program#

# Rules (here they are used such as simple shell scripts)
# --------------------------------------------------------------------------------------------------

compil: mkdir_obj all

mkdir_obj:
	@test -d obj || mkdir obj

clean:
	-rm -r obj *~ Makefile.bak $(PROG) *.i *.o 2> /dev/null || true

# Rules with automatic variables
# --------------------------------------------------------------------------------------------------

all : $(OBJ) $(PROG)

%.o : %.c
	$(CC) -o $@ $(CFLAGS) $^

obj/%.o : %.c
	$(CC) -o $@ $(CFLAGS) -save-temps -fverbose-asm $^

asm: mkdir_obj $(addprefix obj/,$(OBJ) $(FUN))

# Rules for the mains
# --------------------------------------------------------------------------------------------------
scalar_vector_32: scalar_vector_32.o
	$(CC) -o $@ $^ $(LFLAGS)

vector_vector_32: vector_vector_32.o
	$(CC) -o $@ $^ $(LFLAGS)

gather: gather.o
	$(CC) -o $@ $^ $(LFLAGS)
