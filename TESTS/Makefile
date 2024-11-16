include ../make.inc

LIB = ../lib/libgmres.a

.PHONY: all clean

all: test_orth.o test_orth.exe #test_example.o test_example.exe #test_bcgs.o test_bcgs.exe

test_orth.o: test_orth.c
	$(CC) -o $@ -c $^

test_orth.exe: test_orth.o
	$(CC) -o $@ $^ $(LIB) $(LIBS) -lstdc++

test_bcgs.o: test_bcgs.c
	$(CC) -o $@ -c $^

test_bcgs.exe: test_bcgs.o
	$(CC) -o $@ $^ $(SCALAPACK)

test_example.o: test_example.c
	$(CC) -o $@ -c $^

test_example.exe: test_example.o
	$(CC) -o $@ $^ $(SCALAPACK)

clean:
	rm -f *.o *.exe

.SUFFIXES: .o .c
.c.o:
	$(CC) -c $<
