###
# Damien ASSIRE  - 21112838
# Marie BONBOIRE - 21100552
###

# Sources
# --------------------------------------------------------------------------------------------------

OBJ = scalar_vector_32.o dot_product_32.o scalar_vector_mod_32.o
OBJ += profile_sec2_new.o 
OBJ += test_sc_mod.o #test_split.o test_dp.o #test_sc.o

PROG = profile_sec2_new
PROG += test_sc_mod test_split test_dp #test_sc

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
	$(CC) -o $@ $(CFLAGS) -save-temps -fverbose-asm $^

asm: mkdir_obj $(addprefix obj/,$(OBJ) $(FUN))

# Rules for the mains
# --------------------------------------------------------------------------------------------------
test_dp: test_dp.o dot_product_32.o
	$(CC) -o $@ $^ $(LFLAGS)

test_sc_mod: test_sc_mod.o scalar_vector_mod_32.o
	$(CC) -o $@ $^ $(LFLAGS)

test_split: test_split.o
	$(CC) -o $@ $^ $(LFLAGS)

profile_sec2_new: profile_sec2_new.o scalar_vector_32.o dot_product_32.o scalar_vector_mod_32.o
	$(CC) -o $@ $^ $(LFLAGS)