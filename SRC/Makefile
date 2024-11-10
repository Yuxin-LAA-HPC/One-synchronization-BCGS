include ../make.inc

lib = ../lib/libgmres.a

OBJS = orth_1s.o util.o

.PHONY: orth_1s.o util.o

all: $(lib)

orth_1s.o:
	$(CC) -o orth_1s.o -c orth_1s.c

util.o:
	$(CC) -o util.o -c util.c

$(lib): $(OBJS)
	ar cr $@ $(OBJS)

clean:
	rm -f *.o *.exe

.SUFFIXES: .o .c
.c.o:
	$(CC) $(CFLAGS) -c $<
