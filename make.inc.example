BLAS         = -lopenblas
LAPACK       = -llapack
SCALAPACK    = -lscalapack $(LAPACK) $(BLAS) -lgfortran
TPETRA       = -llapack ../tpetra/lib/libkokkostsqr.a ../tpetra/lib/libteuchosnumerics.a ../tpetra/lib/libteuchosremainder.a ../tpetra/lib/libteuchoskokkoscomm.a ../tpetra/lib/libteuchoscomm.a ../tpetra/lib/libteuchoskokkoscompat.a ../tpetra/lib/libteuchosparameterlist.a ../tpetra/lib/libteuchosparser.a ../tpetra/lib/libteuchoscore.a ../tpetra/lib/libkokkoskernels.a ../tpetra/lib/libkokkossimd.a ../tpetra/lib/libkokkosalgorithms.a ../tpetra/lib/libkokkoscontainers.a ../tpetra/lib/libkokkoscore.a  -llapack -lopenblas
LIBS         = $(TPETRA) $(SCALAPACK) -lm -lpthread

#TRILINOSINCLUDE = -I/home/yxma2/package/trilinos/include
TPETRAINCLUDE   = -I/home/yxma2/test_gmres/tpetra/include


CC  = mpicc -O2 -Wall
CXX = mpicxx -O2 -Wall
