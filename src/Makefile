###
# Damien ASSIRE  - 21112838
# Marie BONBOIRE - 21100552
###

# Sources
# --------------------------------------------------------------------------------------------------

OBJ = scalar_vector_32.o vector_vector_32.o gather.o
PROG = scalar_vector_32 vector_vector_32 gather

# Tools and parameters definitions
# --------------------------------------------------------------------------------------------------

CC = gcc

CFLAGS  = -c#                       # Stop after compilation, then produce .o
CFLAGS += -Wall -Wextra -Werror#    # Gives almost all C warnings and considers them to be errors
#CFLAGS += -g#                       # Useful for gdb  FIXME
CFLAGS += -O2#                      # No optimisation mode for compiler  FIXME
CFLAGS += -march=native#            # Enable AVX 
CFLAGS += -Wno-unused-parameter# 
CFLAGS += -Wpedantic
CFLAGS += -Wconversion -Wsign-conversion
CFLAGS += -Wcast-qual -Wcast-align
CFLAGS += -Wundef -Wunused
CFLAGS += -Wdouble-promotion

# to output assembly  FIXME
#CFLAGS += -S -fverbose-asm

LFLAGS  = -fwhole-program#

# Rules (here they are used such as simple shell scripts)
# --------------------------------------------------------------------------------------------------

compil: mkdir_obj all

mkdir_obj:
	test -d obj || mkdir obj

clean:
	-rm -r obj *~ Makefile.bak $(PROG) *.o 2> /dev/null || true

# Rules with automatic variables
# --------------------------------------------------------------------------------------------------

all : $(OBJ) $(PROG)

%.o : %.c
	$(CC) -o $@ $(CFLAGS) $^

# Rules for the mains
# --------------------------------------------------------------------------------------------------
scalar_vector_32: scalar_vector_32.o
	$(CC) -o $@ $^ $(LFLAGS)

vector_vector_32: vector_vector_32.o
	$(CC) -o $@ $^ $(LFLAGS)

gather:gather.o
	$(CC) -o $@ $^ $(LFLAGS)
