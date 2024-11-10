#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "../include/orth.h"
#include "../include/util.h"
#include "../include/lapack.h"

int bcgsi2P1s(int m, int n, int s, double *Xsub, int ldXsub, double *R,
        int ldR, double *work, int lwork)
{
    // Only consider the case m >> n.
    // n >= s0+s;
    if (lwork < 6*n*s + 8*s*s)
    {
        printf("%% Work space is too small. It should be larger than 6*n*s + 8*s*s.\n");
        return 1;
    }

    int myrank_mpi, nprocs_mpi;
    int s0 = 1, p = ceil((double)(n-s0)/(double)s);
    int msub, mres;
    int incx = 1, ntemp, info, s2, mZ, nZ, mScol, nY;
    double normXsub, normX, scal, one = 1.0, zero = 0.0, rone = -1.0;
    double *Scol, *T, *Sdiag, *Ycol, *Omega, *Z, *Ydiag, *P, *Usub;
    double *worktemp;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);

    // Compute the location of Xsub in X.
    msub = m/nprocs_mpi;
    mres = m - msub*nprocs_mpi;
    // Xsub = X(mstart:mstart+msub, :).
    if (mres >= nprocs_mpi - myrank_mpi)
    {
        //mstart = myrank_mpi*msub + mres - nprocs_mpi + myrank_mpi;
        msub = msub + 1;
    }
    //else
    //    mstart = myrank_mpi*msub;

    // Asign spaces for temporary variables.
    // lwork needs to be larger than 6ns+8s^2.
    Scol = work;
    T = Scol + n*s;
    Ycol = T + s*s;
    Omega = Ycol + n*s;
    Z = Omega + n*s;
    P = Z + s*s;
    worktemp = P + s*s;
    Sdiag = worktemp + 2*n*s + 3*s*s;
    Ydiag = Sdiag + s*s;
    Usub = Ydiag + s*s;

    normXsub = dnrm2_(&msub, Xsub, &incx);
    normXsub = normXsub*normXsub;
    MPI_Allreduce(&normXsub, &normX, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    R[0] = sqrt(normX);
    scal = 1.0/R[0];
    // Scale Xsub by 1.0/norm(X(:, 1)) to obtain Qsub(:, 1), which actually
    // stores in  Xsub(:, 1).
    dscal_(&msub, &scal, Xsub, &incx);

    // Update Q2.
    ntemp = s+s0;
    dgemm_("T", "N", &ntemp, &s, &msub, &one, Xsub, &ldXsub,
            &Xsub[s0*ldXsub], &ldXsub, &zero, Scol, &ntemp, 1, 1);
    MPI_Allreduce(Scol, worktemp, ntemp*s, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    dgemm_("T", "N", &s, &s, &s0, &rone, worktemp, &ntemp, worktemp, &ntemp,
            &one, worktemp+s0, &ntemp, 1, 1);
    dlacpy_("A", &s0, &s, worktemp, &ntemp, &R[s0*ldR], &ldR, 1);// Scol stores in R.
    dlacpy_("U", &s, &s, worktemp+s0, &ntemp, Sdiag, &s, 1);
    dpotrf_("U", &s, Sdiag, &s, &info, 1);
    dlacpy_("A", &msub, &s, Xsub, &ldXsub, Usub, &msub, 1);
    dgemm_("N", "N", &msub, &s, &s0, &rone, Usub, &msub, worktemp, &ntemp,
            &one, &Xsub[ldXsub*s0], &ldXsub, 1, 1);
    dtrsm_("R", "U", "N", "N", &msub, &s, &one, Sdiag, &s, &Xsub[ldXsub*s0],
            &ldXsub, 1, 1, 1, 1); // U2 stores in X2.
    nZ = MIN(s, n-s-s0);
    s2 = s + nZ;
    dgemm_("T", "N", &ntemp, &s2, &msub, &one, Xsub, &ldXsub,
            &Xsub[s0*ldXsub], &ldXsub, &zero, Ycol, &ntemp, 1, 1);
    dgemm_("T", "N", &s, &s, &msub, &one, &Xsub[ntemp*ldXsub], &ldXsub,
            &Xsub[ntemp*ldXsub], &ldXsub, &zero, Ycol+ntemp*s2, &s, 1, 1);
    MPI_Allreduce(Ycol, worktemp, ntemp*s2+s*s, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    dgemm_("T", "N", &s, &s, &s0, &rone, worktemp, &ntemp, worktemp, &ntemp,
            &one, worktemp+s0, &ntemp, 1, 1);
    dlacpy_("U", &s, &s, worktemp+s0, &ntemp, Ydiag, &s, 1);
    dpotrf_("U", &s, Ydiag, &s, &info, 1);
    //dlacpy_("A", &msub, &s, Xsub, &ldXsub, Usub, &msub);
    dgemm_("N", "N", &msub, &s, &s0, &rone, Usub, &msub, worktemp, &ntemp,
            &one, &Xsub[ldXsub*s0], &ldXsub, 1, 1);
    dtrsm_("R", "U", "N", "N", &msub, &s, &one, Ydiag, &s, &Xsub[ldXsub*s0],
            &ldXsub, 1, 1, 1, 1); // U2 stores in X2.
    // Update R.
    dgemm_("N", "N", &s0, &s, &s, &one, worktemp, &ntemp, Sdiag, &s, &one,
            &R[s0*ldR], &ldR, 1, 1);
    dgemm_("N", "N", &s, &s, &s, &one, Ydiag, &s, Sdiag, &s, &zero,
            &R[s0*ldR+s0], &ldR, 1, 1);

    mZ = s0;
    nZ = MIN(s, n-s-s0);

    for (int i = 1; i < p; i++)
    {
        mScol = mZ + s;
        dlacpy_("A", &mScol, &nZ, worktemp+ntemp*s, &ntemp, Scol, &mScol, 1);
        dgemm_("T", "N", &s, &nZ, &mZ, &rone, worktemp, &ntemp,
                worktemp+ntemp*s, &ntemp, &one, Scol+mZ, &mScol, 1, 1);
        dtrsm_("L", "U", "T", "N", &s, &nZ, &one, Ydiag, &s, Scol+mZ,
                &mScol, 1, 1, 1, 1);
        dgemm_("T", "N", &nZ, &nZ, &mScol, &rone, Scol, &mScol, Scol,
                &mScol, &one, worktemp+ntemp*s2, &nZ, 1, 1);
        dlacpy_("U", &nZ, &nZ, worktemp+ntemp*s2, &nZ, Sdiag, &nZ, 1);
        dpotrf_("U", &nZ, Sdiag, &nZ, &info, 1);
        dlacpy_("A", &msub, &mScol, Xsub, &ldXsub, Usub, &msub, 1);
        dgemm_("N", "N", &msub, &nZ, &mScol, &rone, Usub, &msub, Scol,
                &mScol, &one, &Xsub[ldXsub*mScol], &ldXsub, 1, 1);
        dtrsm_("R", "U", "N", "N", &msub, &nZ, &one, Sdiag, &nZ,
                &Xsub[ldXsub*mScol], &ldXsub, 1, 1, 1, 1); // U_{k+1} stores in X_{k+1}.
        if (i < p - 1)
        {
            nZ = MIN(s, n-(i+1)*s-s0);
            s2 = s + nZ;
            ntemp = mScol + s;
            dgemm_("T", "N", &ntemp, &s2, &msub, &one, Xsub, &ldXsub,
                    &Xsub[mScol*ldXsub], &ldXsub, &zero, Ycol, &ntemp, 1, 1);
            dgemm_("T", "N", &nZ, &nZ, &msub, &one, &Xsub[ntemp*ldXsub],
                    &ldXsub, &Xsub[ntemp*ldXsub], &ldXsub, &zero,
                    Ycol+ntemp*s2, &nZ, 1, 1);
            MPI_Allreduce(Ycol, worktemp, ntemp*s2+nZ*nZ, MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD);
            nY = s;
        }
        else
        {
            ntemp = mScol + nZ;
            dgemm_("T", "N", &ntemp, &nZ, &msub, &one, Xsub, &ldXsub,
                    &Xsub[mScol*ldXsub], &ldXsub, &zero, Ycol, &ntemp, 1, 1);
            MPI_Allreduce(Ycol, worktemp, ntemp*nZ, MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD);
            nY = nZ;
        }
        dgemm_("T", "N", &nY, &nY, &mScol, &rone, worktemp, &ntemp,
                worktemp, &ntemp, &one, worktemp+mScol, &ntemp, 1, 1);
        dlacpy_("U", &nY, &nY, worktemp+mScol, &ntemp, Ydiag, &nY, 1);
        dpotrf_("U", &nY, Ydiag, &nY, &info, 1);
        //dlacpy_("A", &msub, &s, Xsub, &ldXsub, Usub, &msub);
        dgemm_("N", "N", &msub, &nY, &mScol, &rone, Usub, &msub, worktemp,
                &ntemp, &one, &Xsub[ldXsub*mScol], &ldXsub, 1, 1);
        dtrsm_("R", "U", "N", "N", &msub, &nY, &one, Ydiag, &nY,
                &Xsub[ldXsub*mScol], &ldXsub, 1, 1, 1, 1);
        // Update R.
        dlacpy_("A", &mScol, &nY, Scol, &mScol, &R[mScol*ldR], &ldR, 1);
        dgemm_("N", "N", &mScol, &nY, &nY, &one, worktemp, &ntemp, Sdiag,
                &nY, &one, &R[mScol*ldR], &ldR, 1, 1);
        dgemm_("N", "N", &nY, &nY, &nY, &one, Ydiag, &nY, Sdiag, &nY, &zero,
                &R[mScol*ldR+mScol], &ldR, 1, 1);
        //if (myrank_mpi == 0)
        //{
        //    dprintmat("Sdiag", nY, nY, Sdiag, nY);
        //}
        //MPI_Barrier(MPI_COMM_WORLD);
        //return 0;

        mZ = mZ + s;
    }
    //if (myrank_mpi == 0)
    //    printf("p=%d, mZ=%d, nZ=%d, nY=%d\n", p, mZ, nZ, nY);
    return 0;
}
