#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "../include/lapack.h"
#include "../include/util.h"

void dprintmat(char *name, int nrows, int ncols, double A[], int ldA)
{
    int i, j;

    printf("%s = zeros(%d, %d);\n", name, nrows, ncols);
    for (j = 0; j < ncols; j++)
    for (i = 0; i < nrows; i++)
        printf("%s(%4d, %4d) = %.16e;\n", name, i+1, j+1, A[i + j*ldA]);
    fflush(stdout);
}

void dprintmat_mpi(char *name, int nrows, int ncols, int nrowstart,
        int nrowssub, int ncolssub, double Asub[], int ldAsub)
{
    int i, j, rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        printf("%s = zeros(%d, %d);\n", name, nrows, ncols);
    MPI_Barrier(MPI_COMM_WORLD);
    for (j = 0; j < ncolssub; j++)
    for (i = 0; i < nrowssub; i++)
        printf("%s(%4d, %4d) = %.16e;\n", name, nrowstart+i+1, j+1, Asub[i + j*ldAsub]);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
}

int test_orth_accuracy(int m, int n, int msub, double *Xsub, int ldXsub,
        double *Qsub, int ldQsub, double *R, int ldR,
        double *work, int lwork)
{
    // lwork should be larger than 2*n*n.
    if (lwork < 2*n*n + msub*n)
    {
        printf("%% Work space is too small. It should be larger than 2*n*n+msub*n.\n");
        return 1;
    }
    int myrank_mpi, length = n*n, incx = 1;
    double one = 1.0, zero = 0.0, rone = -1.0, norm, normsub;

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
    dgemm_("T", "N", &n, &n, &msub, &one, Qsub, &ldQsub, Qsub, &ldQsub,
            &zero, work, &n, 1, 1);
    MPI_Allreduce(work, work+n*n, n*n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (myrank_mpi == 0)
    {
        for (int i = 0; i < n; i++)
            work[n*n+i*n+i] = work[n*n+i*n+i] - 1.0;
        norm = dnrm2_(&length, work+n*n, &incx);
        printf("%% Loss of orthogonality (||Q'*Q-I||): %.18f\n", norm);
    }

    dgemm_("N", "N", &msub, &n, &n, &one, Qsub, &ldQsub, R, &ldR, &rone,
            Xsub, &ldXsub, 1, 1);
    dlacpy_("A", &msub, &n, Xsub, &ldXsub, work, &msub, 1);
    length = msub*n;
    normsub = dnrm2_(&length, work, &incx);
    normsub = normsub*normsub;
    MPI_Allreduce(&normsub, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    norm = sqrt(norm);
    if (myrank_mpi == 0)
    {
        printf("%% Backward error (||QR-X||): %.18f\n", norm);
    }

    return 0;
}
