include ../make.inc

lib = ../lib/libgmres.a

OBJS = orth.o util.o

.PHONY: orth.o util.o intraorth.o

all: $(lib)

orth.o:
	$(CXX) $(TPETRAINCLUDE) -o orth.o -c orth.c

intraorth.o:
	$(CXX) $(TPETRAINCLUDE) -o intraorth.o -c intraorth.c

util.o:
	$(CC) -o util.o -c util.c

$(lib): $(OBJS)
	ar cr $@ $(OBJS)

clean:
	rm -f *.o *.exe

.SUFFIXES: .o .c
.c.o:
	$(CC) $(CFLAGS) -c $<
