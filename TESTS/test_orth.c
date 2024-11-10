#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "../include/util.h"
#include "../include/orth.h"
#include "../include/lapack.h"
#include "../include/scalapack.h"


int test_orth_1s(int m, int n, int s)
{
    int myrank_mpi, nprocs_mpi, rank;
    int msub, mres, mstart, lwork = 16*n*s + 8*s*s;
    double *Xsub, *Qsub;
    double *R = (double *)calloc(n*n, sizeof(double));
    double *work = (double *)calloc(lwork, sizeof(double));
    double starttime, mytime, time;

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);

    msub = m/nprocs_mpi;
    mres = m - msub*nprocs_mpi;
    if (mres >= nprocs_mpi - myrank_mpi)
    {
        mstart = myrank_mpi*msub + mres - nprocs_mpi + myrank_mpi;
        msub = msub + 1;
    }
    else
        mstart = myrank_mpi*msub;
    //printf("rank=%d, msub=%d, mstart=%d\n", myrank_mpi, msub, mstart);
    //MPI_Barrier(MPI_COMM_WORLD);

    // Generate the test matrix X.
    Xsub = (double *)calloc(msub*n, sizeof(double));
    Qsub = (double *)calloc(msub*n, sizeof(double));
    srand(myrank_mpi);
    for (int i = 0; i < msub*n; i++)
        Xsub[i] = rand()%100;
    dlacpy_("A", &msub, &n, Xsub, &msub, Qsub, &msub, 1);
    //dprintmat_mpi("X", m, n, mstart, msub, n, Xsub, msub);

    starttime = MPI_Wtime();
    bcgsi2P1s(m, n, s, Qsub, msub, R, n, work, lwork);
    mytime = MPI_Wtime() - starttime;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&mytime, &time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        printf("%% Test time=%f\n", time);
        //dprintmat("R", n, n, R, n);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //dprintmat_mpi("Q", m, n, mstart, msub, n, Qsub, msub);
    //MPI_Barrier(MPI_COMM_WORLD);

    test_orth_accuracy(m, n, msub, Xsub, msub, Qsub, msub, R, n, work, lwork);

    free(Xsub);
    free(Qsub);
    free(R);
    free(work);
    return 0;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    test_orth_1s(13, 10, 2);
    MPI_Finalize();
    return 0;
}
