BLAS         = -lopenblas
LAPACK       = -llapack
SCALAPACK    = -lscalapack $(LAPACK) $(BLAS) -lgfortran
LIBS         = $(SCALAPACK) -lm -lpthread

CC = mpicc -O2 -Wall
