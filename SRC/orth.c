#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "../include/orth.h"
#include "../include/util.h"
#include "../include/lapack.h"
#include "../tpetra/include/Tsqr.hpp"
#include "../tpetra/include/Tsqr_TeuchosMessenger.hpp"
#include "../tpetra/include/Teuchos_DefaultMpiComm.hpp"
#include "../tpetra/include/Tsqr_NodeTsqrFactory.hpp"
#include "../tpetra/include/Tsqr_Random_NormalGenerator.hpp"

TSQR::Tsqr<int, double> init_tsqr()
{
    using node_tsqr_type = TSQR::NodeTsqr<int, double>;
    using dist_tsqr_type = TSQR::DistTsqr<int, double>;
    Teuchos::RCP<dist_tsqr_type> dist_tsqr(new dist_tsqr_type);
    Teuchos::RCP<node_tsqr_type> node_tsqr;
    std::string nodeTsqrName("Default");

    // Set node_tsqr.
    using device_type = Kokkos::DefaultExecutionSpace::device_type;
    using node_tsqr_factory_type = TSQR::NodeTsqrFactory<
          double, int, device_type>;
    node_tsqr = node_tsqr_factory_type::getNodeTsqr(nodeTsqrName);

    // Set dist_tsqr.
    using Teuchos::rcp_implicit_cast;
    using Teuchos::rcp;
    using Teuchos::RCP;
    using comm_type = Teuchos::MpiComm<int>;
    using messenger_type = TSQR::TeuchosMessenger<double>;
    const auto comm = Teuchos::rcp(new comm_type(MPI_COMM_WORLD));
    messenger_type teuchos_messenger(comm);
    auto scalarMess = rcp(new TSQR::TeuchosMessenger<double>(comm));
    auto scalarMessBase =
          rcp_implicit_cast<TSQR::MessengerBase<double>>(scalarMess);
    dist_tsqr->init(scalarMessBase);

    using tsqr_type = TSQR::Tsqr<int, double>;
    tsqr_type mytsqr(node_tsqr, dist_tsqr);
    //mytsqr.factorExplicit(msub, s0, Xsub, ldXsub, Qsub, ldQsub, R, ldR, false);

    return mytsqr;
}

int bcgsi2P1s(int m, int n, int s, double *Xsub, int ldXsub, double *R,
        int ldR, double *work, int lwork)
{
    // Only consider the case m >> n.
    // n >= s0+s;

    int myrank_mpi, nprocs_mpi;
    int s0 = s, p = ceil((double)(n-s0)/(double)s);
    int msub, mres;
    int ntemp, info, s2, mZ, nZ, mScol, nY;
    //int incx = 1;
    //double normXsub, normX, scal;
    double one = 1.0, zero = 0.0, rone = -1.0;
    double *Scol, *T, *Sdiag, *Ycol, *Ydiag, *Usub;
    double *worktemp;
    char TRANS[] = "T", NOTRANS[] = "N", ALL[] = "A", UPPER[] = "U";
    char LOWER[] = "L", LEFT[] = "L", RIGHT[] = "R", NONUNIT[] = "N";
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
    if (lwork < 5*n*s + 9*s*s + msub*n)
    {
        printf("%% Work space is too small. It should be larger than 6*n*s + 8*s*s.\n");
        return 1;
    }
    Scol = work;
    T = Scol + n*s;
    Ycol = T + s*s;
    worktemp = Ycol + 2*n*s + 3*s*s;
    Sdiag = worktemp + 2*n*s + 3*s*s;
    Ydiag = Sdiag + s*s;
    Usub = Ydiag + s*s;

    //normXsub = dnrm2_(&msub, Xsub, &incx);
    //normXsub = normXsub*normXsub;
    //MPI_Allreduce(&normXsub, &normX, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //R[0] = sqrt(normX);
    //scal = 1.0/R[0];
    //// Scale Xsub by 1.0/norm(X(:, 1)) to obtain Qsub(:, 1), which actually
    //// stores in  Xsub(:, 1).
    //dscal_(&msub, &scal, Xsub, &incx);

    // Perform intraorthogonalization to the first block column.
    dlacpy_(ALL, &msub, &s, Xsub, &ldXsub, Usub, &msub, 1);
    TSQR::Tsqr<int, double> mytsqr = init_tsqr();
    mytsqr.factorExplicit(msub, s0, Usub, msub, Xsub, ldXsub, R, ldR, false);

    // Update Q2.
    ntemp = s+s0;
    dgemm_(TRANS, NOTRANS, &ntemp, &s, &msub, &one, Xsub, &ldXsub,
            &Xsub[s0*ldXsub], &ldXsub, &zero, Scol, &ntemp, 1, 1);
    MPI_Allreduce(Scol, worktemp, ntemp*s, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    dgemm_(TRANS, NOTRANS, &s, &s, &s0, &rone, worktemp, &ntemp, worktemp,
            &ntemp, &one, worktemp+s0, &ntemp, 1, 1);
    dlacpy_(ALL, &s0, &s, worktemp, &ntemp, &R[s0*ldR], &ldR, 1);// Scol stores in R.
    dlacpy_(UPPER, &s, &s, worktemp+s0, &ntemp, Sdiag, &s, 1);
    dpotrf_(UPPER, &s, Sdiag, &s, &info, 1);
    //dlacpy_(ALL, &msub, &s, Xsub, &ldXsub, Usub, &msub, 1);
    dgemm_(NOTRANS, NOTRANS, &msub, &s, &s0, &rone, Xsub, &ldXsub, worktemp,
            &ntemp, &one, &Xsub[ldXsub*s0], &ldXsub, 1, 1);
    dtrsm_(RIGHT, UPPER, NOTRANS, NOTRANS, &msub, &s, &one, Sdiag, &s,
            &Xsub[ldXsub*s0], &ldXsub, 1, 1, 1, 1); // U2 stores in X2.
    nZ = MIN(s, n-s-s0);
    s2 = s + nZ;
    dgemm_(TRANS, NOTRANS, &ntemp, &s2, &msub, &one, Xsub, &ldXsub,
            &Xsub[s0*ldXsub], &ldXsub, &zero, Ycol, &ntemp, 1, 1);
    dgemm_(TRANS, NOTRANS, &s, &s, &msub, &one, &Xsub[ntemp*ldXsub],
            &ldXsub, &Xsub[ntemp*ldXsub], &ldXsub, &zero, Ycol+ntemp*s2,
            &s, 1, 1);
    MPI_Allreduce(Ycol, worktemp, ntemp*s2+s*s, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    dgemm_(TRANS, NOTRANS, &s, &s, &s0, &rone, worktemp, &ntemp, worktemp,
            &ntemp, &one, worktemp+s0, &ntemp, 1, 1);
    dlacpy_(UPPER, &s, &s, worktemp+s0, &ntemp, Ydiag, &s, 1);
    dpotrf_(UPPER, &s, Ydiag, &s, &info, 1);
    //dlacpy_("A", &msub, &s, Xsub, &ldXsub, Usub, &msub);
    dgemm_(NOTRANS, NOTRANS, &msub, &s, &s0, &rone, Xsub, &ldXsub, worktemp,
            &ntemp, &one, &Xsub[ldXsub*s0], &ldXsub, 1, 1);
    dtrsm_(RIGHT, UPPER, NOTRANS, NONUNIT, &msub, &s, &one, Ydiag, &s,
            &Xsub[ldXsub*s0], &ldXsub, 1, 1, 1, 1); // U2 stores in X2.
    // Update R.
    dgemm_(NOTRANS, NOTRANS, &s0, &s, &s, &one, worktemp, &ntemp, Sdiag,
            &s, &one, &R[s0*ldR], &ldR, 1, 1);
    dgemm_(NOTRANS, NOTRANS, &s, &s, &s, &one, Ydiag, &s, Sdiag, &s, &zero,
            &R[s0*ldR+s0], &ldR, 1, 1);

    mZ = s0;
    nZ = MIN(s, n-s-s0);

    for (int i = 1; i < p; i++)
    {
        mScol = mZ + s;
        dlacpy_(ALL, &mScol, &nZ, worktemp+ntemp*s, &ntemp, Scol, &mScol, 1);
        dgemm_(TRANS, NOTRANS, &s, &nZ, &mZ, &rone, worktemp, &ntemp,
                worktemp+ntemp*s, &ntemp, &one, Scol+mZ, &mScol, 1, 1);
        dtrsm_(LEFT, UPPER, TRANS, NONUNIT, &s, &nZ, &one, Ydiag, &s,
                Scol+mZ, &mScol, 1, 1, 1, 1);
        dgemm_(TRANS, NOTRANS, &nZ, &nZ, &mScol, &rone, Scol, &mScol, Scol,
                &mScol, &one, worktemp+ntemp*s2, &nZ, 1, 1);
        dlaset_(LOWER, &nZ, &nZ, &zero, &zero, Sdiag, &nZ, 1);
        dlacpy_(UPPER, &nZ, &nZ, worktemp+ntemp*s2, &nZ, Sdiag, &nZ, 1);
        //if (myrank_mpi == 0)
        //{
        //    printf("mScol=%d, nY=%d, nZ=%d\n", mScol, nY, nZ);
        //    dprintmat("Sdiag", nZ, nZ, Sdiag, nZ);
        //    dprintmat("Ydiag", nZ, nZ, Ydiag, nZ);
        //}
        //MPI_Barrier(MPI_COMM_WORLD);
        dpotrf_(UPPER, &nZ, Sdiag, &nZ, &info, 1);
        //dlacpy_(ALL, &msub, &mScol, Xsub, &ldXsub, Usub, &msub, 1);
        dgemm_(NOTRANS, NOTRANS, &msub, &nZ, &mScol, &rone, Xsub, &ldXsub,
                Scol, &mScol, &one, &Xsub[ldXsub*mScol], &ldXsub, 1, 1);
        dtrsm_(RIGHT, UPPER, NOTRANS, NONUNIT, &msub, &nZ, &one, Sdiag, &nZ,
                &Xsub[ldXsub*mScol], &ldXsub, 1, 1, 1, 1); // U_{k+1} stores in X_{k+1}.
        if (i < p - 1)
        {
            nZ = MIN(s, n-(i+1)*s-s0);
            s2 = s + nZ;
            ntemp = mScol + s;
            dgemm_(TRANS, NOTRANS, &ntemp, &s2, &msub, &one, Xsub, &ldXsub,
                    &Xsub[mScol*ldXsub], &ldXsub, &zero, Ycol, &ntemp, 1, 1);
            dgemm_(TRANS, NOTRANS, &nZ, &nZ, &msub, &one, &Xsub[ntemp*ldXsub],
                    &ldXsub, &Xsub[ntemp*ldXsub], &ldXsub, &zero,
                    Ycol+ntemp*s2, &nZ, 1, 1);
            MPI_Allreduce(Ycol, worktemp, ntemp*s2+nZ*nZ, MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD);
            nY = s;
        }
        else
        {
            ntemp = mScol + nZ;
            dgemm_(TRANS, NOTRANS, &ntemp, &nZ, &msub, &one, Xsub, &ldXsub,
                    &Xsub[mScol*ldXsub], &ldXsub, &zero, Ycol, &ntemp, 1, 1);
            MPI_Allreduce(Ycol, worktemp, ntemp*nZ, MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD);
            nY = nZ;
        }
        dgemm_(TRANS, NOTRANS, &nY, &nY, &mScol, &rone, worktemp, &ntemp,
                worktemp, &ntemp, &one, worktemp+mScol, &ntemp, 1, 1);
        dlaset_(LOWER, &nY, &nY, &zero, &zero, Ydiag, &nY, 1);
        dlacpy_(UPPER, &nY, &nY, worktemp+mScol, &ntemp, Ydiag, &nY, 1);
        dpotrf_(UPPER, &nY, Ydiag, &nY, &info, 1);
        //dlacpy_("A", &msub, &s, Xsub, &ldXsub, Usub, &msub);
        dgemm_(NOTRANS, NOTRANS, &msub, &nY, &mScol, &rone, Xsub, &ldXsub,
                worktemp, &ntemp, &one, &Xsub[ldXsub*mScol], &ldXsub, 1, 1);
        dtrsm_(RIGHT, UPPER, NOTRANS, NONUNIT, &msub, &nY, &one, Ydiag, &nY,
                &Xsub[ldXsub*mScol], &ldXsub, 1, 1, 1, 1);
        // Update R.
        dlacpy_(ALL, &mScol, &nY, Scol, &mScol, &R[mScol*ldR], &ldR, 1);
        dgemm_(NOTRANS, NOTRANS, &mScol, &nY, &nY, &one, worktemp, &ntemp,
                Sdiag, &nY, &one, &R[mScol*ldR], &ldR, 1, 1);
        dgemm_(NOTRANS, NOTRANS, &nY, &nY, &nY, &one, Ydiag, &nY, Sdiag,
                &nY, &zero, &R[mScol*ldR+mScol], &ldR, 1, 1);

        mZ = mZ + s;
    }
    //if (myrank_mpi == 0)
    //    printf("p=%d, mZ=%d, nZ=%d, nY=%d\n", p, mZ, nZ, nY);
    return 0;
}

int bcgsi21s(int m, int n, int s, double *Xsub, int ldXsub, double *R,
        int ldR, double *work, int lwork)
{
    // Only consider the case m >> n.
    // n >= s0+s;

    int myrank_mpi, nprocs_mpi;
    int s0 = s, p = ceil((double)(n-s0)/(double)s);
    int msub, mres;
    int ntemp, info, s2, mZ, nZ, mScol, nY, length;
    int incx = 1;
    //double normXsub, normX, scal;
    double one = 1.0, zero = 0.0, rone = -1.0;
    double *Scol, *Sdiag, *Ycol, *Ydiag, *Usub;
    double *worktemp;
    char TRANS[] = "T", NOTRANS[] = "N", ALL[] = "A", UPPER[] = "U";
    char LOWER[] = "L", LEFT[] = "L", RIGHT[] = "R", NONUNIT[] = "N";
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);

    // Compute the location of Xsub in X.
    msub = m/nprocs_mpi;
    mres = m - msub*nprocs_mpi;
    if (mres >= nprocs_mpi - myrank_mpi)
    {
        //mstart = myrank_mpi*msub + mres - nprocs_mpi + myrank_mpi;
        msub = msub + 1;
    }

    // Asign spaces for temporary variables.
    // lwork needs to be larger than 6ns+8s^2.
    if (lwork < 5*n*s + 9*s*s + msub*n)
    {
        printf("%% Work space is too small. It should be larger than 6*n*s + 8*s*s.\n");
        return 1;
    }
    Scol = work;
    Ycol = Scol + n*s;
    worktemp = Ycol + 2*n*s + 3*s*s;
    Sdiag = worktemp + 2*n*s + 3*s*s;
    Ydiag = Sdiag + s*s;
    Usub = Ydiag + s*s;


    // Perform intraorthogonalization to the first block column.
    dlacpy_(ALL, &msub, &s, Xsub, &ldXsub, Usub, &msub, 1);
    TSQR::Tsqr<int, double> mytsqr = init_tsqr();
    mytsqr.factorExplicit(msub, s0, Usub, msub, Xsub, ldXsub, R, ldR, false);

    // Update Q2.
    ntemp = s+s0;
    dgemm_(TRANS, NOTRANS, &s0, &s, &msub, &one, Xsub, &ldXsub,
            &Xsub[s0*ldXsub], &ldXsub, &zero, worktemp, &s0, 1, 1);
    MPI_Allreduce(worktemp, Scol, s0*s, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    //dlacpy_(ALL, &msub, &s, Xsub, &ldXsub, Usub, &msub, 1);
    dgemm_(NOTRANS, NOTRANS, &msub, &s, &s0, &rone, Xsub, &ldXsub, Scol,
            &s0, &one, &Xsub[ldXsub*s0], &ldXsub, 1, 1);
    nZ = MIN(s, n-s-s0);
    s2 = s + nZ;
    dgemm_(TRANS, NOTRANS, &ntemp, &s2, &msub, &one, Xsub, &ldXsub,
            &Xsub[s0*ldXsub], &ldXsub, &zero, Ycol, &ntemp, 1, 1);
    MPI_Allreduce(Ycol, worktemp, ntemp*s2, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    dgemm_(TRANS, NOTRANS, &s, &s, &s0, &rone, worktemp, &ntemp, worktemp,
            &ntemp, &one, worktemp+s0, &ntemp, 1, 1);
    dlacpy_(UPPER, &s, &s, worktemp+s0, &ntemp, Ydiag, &s, 1);
    dpotrf_(UPPER, &s, Ydiag, &s, &info, 1);
    //dlacpy_("A", &msub, &s, Xsub, &ldXsub, Usub, &msub);
    dgemm_(NOTRANS, NOTRANS, &msub, &s, &s0, &rone, Xsub, &ldXsub, worktemp,
            &ntemp, &one, &Xsub[ldXsub*s0], &ldXsub, 1, 1);
    dtrsm_(RIGHT, UPPER, NOTRANS, NONUNIT, &msub, &s, &one, Ydiag, &s,
            &Xsub[ldXsub*s0], &ldXsub, 1, 1, 1, 1); // U2 stores in X2.
    // Update R.
    length = s0*s;
    dlacpy_(ALL, &s0, &s, worktemp, &ntemp, Ycol, &s0, 1);
    daxpy_(&length, &one, Ycol, &incx, Scol, &incx);
    dlacpy_(ALL, &s0, &s, Scol, &s0, &R[s0*ldR], &ldR, 1);
    dlacpy_(UPPER, &s, &s, Ydiag, &s, &R[s0*ldR+s0], &ldR, 1);

    mZ = s0;
    nZ = MIN(s, n-s-s0);

    for (int i = 1; i < p; i++)
    {
        mScol = mZ + s;
        dlacpy_(ALL, &mScol, &nZ, worktemp+ntemp*s, &ntemp, Scol, &mScol, 1);
        dgemm_(TRANS, NOTRANS, &s, &nZ, &mZ, &rone, worktemp, &ntemp,
                worktemp+ntemp*s, &ntemp, &one, Scol+mZ, &mScol, 1, 1);
        dtrsm_(LEFT, UPPER, TRANS, NONUNIT, &s, &nZ, &one, Ydiag, &s,
                Scol+mZ, &mScol, 1, 1, 1, 1);
        //if (myrank_mpi == 0)
        //{
        //    printf("mScol=%d, nY=%d, nZ=%d\n", mScol, nY, nZ);
        //    dprintmat("Sdiag", nZ, nZ, Sdiag, nZ);
        //    dprintmat("Ydiag", nZ, nZ, Ydiag, nZ);
        //}
        //MPI_Barrier(MPI_COMM_WORLD);
        //dlacpy_(ALL, &msub, &mScol, Xsub, &ldXsub, Usub, &msub, 1);
        dgemm_(NOTRANS, NOTRANS, &msub, &nZ, &mScol, &rone, Xsub, &ldXsub,
                Scol, &mScol, &one, &Xsub[ldXsub*mScol], &ldXsub, 1, 1);
        if (i < p - 1)
        {
            nZ = MIN(s, n-(i+1)*s-s0);
            s2 = s + nZ;
            ntemp = mScol + s;
            dgemm_(TRANS, NOTRANS, &ntemp, &s2, &msub, &one, Xsub, &ldXsub,
                    &Xsub[mScol*ldXsub], &ldXsub, &zero, Ycol, &ntemp, 1, 1);
            MPI_Allreduce(Ycol, worktemp, ntemp*s2, MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD);
            nY = s;
        }
        else
        {
            ntemp = mScol + nZ;
            dgemm_(TRANS, NOTRANS, &ntemp, &nZ, &msub, &one, Xsub, &ldXsub,
                    &Xsub[mScol*ldXsub], &ldXsub, &zero, Ycol, &ntemp, 1, 1);
            MPI_Allreduce(Ycol, worktemp, ntemp*nZ, MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD);
            nY = nZ;
        }
        dgemm_(TRANS, NOTRANS, &nY, &nY, &mScol, &rone, worktemp, &ntemp,
                worktemp, &ntemp, &one, worktemp+mScol, &ntemp, 1, 1);
        dlaset_(LOWER, &nY, &nY, &zero, &zero, Ydiag, &nY, 1);
        dlacpy_(UPPER, &nY, &nY, worktemp+mScol, &ntemp, Ydiag, &nY, 1);
        dpotrf_(UPPER, &nY, Ydiag, &nY, &info, 1);
        //dlacpy_("A", &msub, &s, Xsub, &ldXsub, Usub, &msub);
        dgemm_(NOTRANS, NOTRANS, &msub, &nY, &mScol, &rone, Xsub, &ldXsub,
                worktemp, &ntemp, &one, &Xsub[ldXsub*mScol], &ldXsub, 1, 1);
        dtrsm_(RIGHT, UPPER, NOTRANS, NONUNIT, &msub, &nY, &one, Ydiag, &nY,
                &Xsub[ldXsub*mScol], &ldXsub, 1, 1, 1, 1);
        // Update R.
        length = mScol*nY;
        dlacpy_(ALL, &mScol, &nY, worktemp, &ntemp, Ycol, &mScol, 1);
        daxpy_(&length, &one, Ycol, &incx, Scol, &incx);
        dlacpy_(ALL, &mScol, &nY, Scol, &mScol, &R[mScol*ldR], &ldR, 1);
        dlacpy_(UPPER, &nY, &nY, Ydiag, &nY, &R[mScol*ldR+mScol], &ldR, 1);

        mZ = mZ + s;
    }
    //if (myrank_mpi == 0)
    //    printf("p=%d, mZ=%d, nZ=%d, nY=%d\n", p, mZ, nZ, nY);
    return 0;
}

int bcgspipi2(int m, int n, int s, double *Xsub, int ldXsub, double *R,
        int ldR, double *work, int lwork)
{
    // Only consider the case m >> n.
    // n >= s0+s;

    int myrank_mpi, nprocs_mpi;
    int s0 = s, p = ceil((double)(n-s0)/(double)s);
    int msub, mres;
    int ntemp, info, nZ, mScol, nY;
    double one = 1.0, zero = 0.0, rone = -1.0;
    double *Scol, *Sdiag, *Ycol, *Ydiag, *Usub;
    double *worktemp;
    char TRANS[] = "T", NOTRANS[] = "N", ALL[] = "A", UPPER[] = "U";
    char LOWER[] = "L", RIGHT[] = "R", NONUNIT[] = "N";
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);

    // Compute the location of Xsub in X.
    msub = m/nprocs_mpi;
    mres = m - msub*nprocs_mpi;
    if (mres >= nprocs_mpi - myrank_mpi)
    {
        msub = msub + 1;
    }

    // Asign spaces for temporary variables.
    // lwork needs to be larger than 6ns+8s^2.
    if (lwork < 5*n*s + 9*s*s + msub*n)
    {
        printf("%% Work space is too small. It should be larger than 6*n*s + 8*s*s.\n");
        return 1;
    }
    Scol = work;
    Ycol = Scol + n*s;
    worktemp = Ycol + 2*n*s + 3*s*s;
    Sdiag = worktemp + 2*n*s + 3*s*s;
    Ydiag = Sdiag + s*s;
    Usub = Ydiag + s*s;


    // Perform intraorthogonalization to the first block column.
    dlacpy_(ALL, &msub, &s, Xsub, &ldXsub, Usub, &msub, 1);
    TSQR::Tsqr<int, double> mytsqr = init_tsqr();
    mytsqr.factorExplicit(msub, s0, Usub, msub, Xsub, ldXsub, R, ldR, false);

    mScol = s0;

    for (int i = 0; i < p; i++)
    {
        nZ = MIN(s, n-i*s-s0);
        ntemp = mScol + nZ;
        dgemm_(TRANS, NOTRANS, &ntemp, &nZ, &msub, &one, Xsub, &ldXsub,
                &Xsub[mScol*ldXsub], &ldXsub, &zero, worktemp, &ntemp, 1, 1);
        MPI_Allreduce(worktemp, Scol, ntemp*nZ, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
        dgemm_(TRANS, NOTRANS, &nZ, &nZ, &mScol, &rone, Scol, &ntemp, Scol,
                &ntemp, &one, Scol+mScol, &ntemp, 1, 1);
        dlaset_(LOWER, &nZ, &nZ, &zero, &zero, Sdiag, &nZ, 1);
        dlacpy_(UPPER, &nZ, &nZ, Scol+mScol, &ntemp, Sdiag, &nZ, 1);
        dpotrf_(UPPER, &nZ, Sdiag, &nZ, &info, 1);
        //dlacpy_(ALL, &msub, &mScol, Xsub, &ldXsub, Usub, &msub, 1);
        dgemm_(NOTRANS, NOTRANS, &msub, &nZ, &mScol, &rone, Xsub, &ldXsub,
                Scol, &ntemp, &one, &Xsub[ldXsub*mScol], &ldXsub, 1, 1);
        dtrsm_(RIGHT, UPPER, NOTRANS, NONUNIT, &msub, &nZ, &one, Sdiag, &nZ,
                &Xsub[ldXsub*mScol], &ldXsub, 1, 1, 1, 1); // U_{k+1} stores in X_{k+1}.
        dgemm_(TRANS, NOTRANS, &ntemp, &nZ, &msub, &one, Xsub, &ldXsub,
                &Xsub[mScol*ldXsub], &ldXsub, &zero, Ycol, &ntemp, 1, 1);
        MPI_Allreduce(Ycol, worktemp, ntemp*nZ, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD);
        nY = nZ;
        dgemm_(TRANS, NOTRANS, &nY, &nY, &mScol, &rone, worktemp, &ntemp,
                worktemp, &ntemp, &one, worktemp+mScol, &ntemp, 1, 1);
        dlaset_(LOWER, &nY, &nY, &zero, &zero, Ydiag, &nY, 1);
        dlacpy_(UPPER, &nY, &nY, worktemp+mScol, &ntemp, Ydiag, &nY, 1);
        dpotrf_(UPPER, &nY, Ydiag, &nY, &info, 1);
        //dlacpy_("A", &msub, &s, Xsub, &ldXsub, Usub, &msub);
        dgemm_(NOTRANS, NOTRANS, &msub, &nY, &mScol, &rone, Xsub, &ldXsub,
                worktemp, &ntemp, &one, &Xsub[ldXsub*mScol], &ldXsub, 1, 1);
        dtrsm_(RIGHT, UPPER, NOTRANS, NONUNIT, &msub, &nY, &one, Ydiag, &nY,
                &Xsub[ldXsub*mScol], &ldXsub, 1, 1, 1, 1);
        // Update R.
        dlacpy_(ALL, &mScol, &nY, Scol, &ntemp, &R[mScol*ldR], &ldR, 1);
        dgemm_(NOTRANS, NOTRANS, &mScol, &nY, &nY, &one, worktemp, &ntemp,
                Sdiag, &nY, &one, &R[mScol*ldR], &ldR, 1, 1);
        dgemm_(NOTRANS, NOTRANS, &nY, &nY, &nY, &one, Ydiag, &nY, Sdiag,
                &nY, &zero, &R[mScol*ldR+mScol], &ldR, 1, 1);

        mScol = mScol + s;
    }
    //if (myrank_mpi == 0)
    //    printf("p=%d, mZ=%d, nZ=%d, nY=%d\n", p, mZ, nZ, nY);
    return 0;
}

int bcgsi2(int m, int n, int s, double *Xsub, int ldXsub, double *R,
        int ldR, double *work, int lwork)
{
    // Only consider the case m >> n.
    // n >= s0+s;

    int myrank_mpi, nprocs_mpi;
    int s0 = s, p = ceil((double)(n-s0)/(double)s);
    int msub, mres;
    int nZ, mScol, nY;
    double one = 1.0, zero = 0.0, rone = -1.0;
    double *Scol, *Sdiag, *Ycol, *Ydiag, *Usub;
    double *worktemp;
    char TRANS[] = "T", NOTRANS[] = "N", ALL[] = "A";
    char LOWER[] = "L";
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);

    // Compute the location of Xsub in X.
    msub = m/nprocs_mpi;
    mres = m - msub*nprocs_mpi;
    if (mres >= nprocs_mpi - myrank_mpi) {msub = msub + 1;}

    // Asign spaces for temporary variables.
    // lwork needs to be larger than 6ns+8s^2.
    if (lwork < 5*n*s + 9*s*s + msub*s)
    {
        printf("%% Work space is too small. It should be larger than 6*n*s + 8*s*s.\n");
        return 1;
    }
    Scol = work;
    Ycol = Scol + n*s;
    worktemp = Ycol + 2*n*s + 3*s*s;
    Sdiag = worktemp + 2*n*s + 3*s*s;
    Ydiag = Sdiag + s*s;
    Usub = Ydiag + s*s;


    // Perform intraorthogonalization to the first block column.
    dlacpy_(ALL, &msub, &s, Xsub, &ldXsub, Usub, &msub, 1);
    TSQR::Tsqr<int, double> mytsqr = init_tsqr();
    mytsqr.factorExplicit(msub, s0, Usub, msub, Xsub, ldXsub, R, ldR, false);

    mScol = s0;

    for (int i = 0; i < p; i++)
    {
        nZ = MIN(s, n-i*s-s0);
        dgemm_(TRANS, NOTRANS, &mScol, &nZ, &msub, &one, Xsub, &ldXsub,
                &Xsub[mScol*ldXsub], &ldXsub, &zero, worktemp, &mScol, 1, 1);
        MPI_Allreduce(worktemp, Scol, mScol*nZ, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
        dlaset_(LOWER, &nZ, &nZ, &zero, &zero, Sdiag, &nZ, 1);
        //dlacpy_(ALL, &msub, &mScol, Xsub, &ldXsub, Usub, &msub, 1);
        dgemm_(NOTRANS, NOTRANS, &msub, &nZ, &mScol, &rone, Xsub, &ldXsub,
                Scol, &mScol, &one, &Xsub[ldXsub*mScol], &ldXsub, 1, 1);
        dlacpy_(ALL, &msub, &nZ, &Xsub[ldXsub*mScol], &ldXsub, Usub,
                &msub, 1);
        mytsqr.factorExplicit(msub, nZ, Usub, msub, &Xsub[ldXsub*mScol],
                ldXsub, Sdiag, nZ, false);
        dgemm_(TRANS, NOTRANS, &mScol, &nZ, &msub, &one, Xsub, &ldXsub,
                &Xsub[mScol*ldXsub], &ldXsub, &zero, Ycol, &mScol, 1, 1);
        MPI_Allreduce(Ycol, worktemp, mScol*nZ, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD);
        nY = nZ;
        dlaset_(LOWER, &nY, &nY, &zero, &zero, Ydiag, &nY, 1);
        dgemm_(NOTRANS, NOTRANS, &msub, &nY, &mScol, &rone, Xsub, &ldXsub,
                worktemp, &mScol, &one, &Xsub[ldXsub*mScol], &ldXsub, 1, 1);
        dlacpy_(ALL, &msub, &nZ, &Xsub[ldXsub*mScol], &ldXsub, Usub,
                &msub, 1);
        mytsqr.factorExplicit(msub, nZ, Usub, msub, &Xsub[ldXsub*mScol],
                ldXsub, Ydiag, nZ, false);
        // Update R.
        dlacpy_(ALL, &mScol, &nY, Scol, &mScol, &R[mScol*ldR], &ldR, 1);
        dgemm_(NOTRANS, NOTRANS, &mScol, &nY, &nY, &one, worktemp, &mScol,
                Sdiag, &nY, &one, &R[mScol*ldR], &ldR, 1, 1);
        dgemm_(NOTRANS, NOTRANS, &nY, &nY, &nY, &one, Ydiag, &nY, Sdiag,
                &nY, &zero, &R[mScol*ldR+mScol], &ldR, 1, 1);

        mScol = mScol + s;
    }
    //if (myrank_mpi == 0)
    //    printf("p=%d, mZ=%d, nZ=%d, nY=%d\n", p, mZ, nZ, nY);
    return 0;
}

int bcgsi2P2s(int m, int n, int s, double *Xsub, int ldXsub, double *R,
        int ldR, double *work, int lwork)
{
    // Only consider the case m >> n.
    // n >= s0+s;

    int myrank_mpi, nprocs_mpi;
    int s0 = s, p = ceil((double)(n-s0)/(double)s);
    int msub, mres;
    int ntemp, info, s2, mZ, nZ, mScol, nY;
    //int incx = 1;
    //double normXsub, normX, scal;
    double one = 1.0, zero = 0.0, rone = -1.0;
    double *Scol, *T, *Sdiag, *Ycol, *Ydiag, *Usub;
    double *worktemp;
    char TRANS[] = "T", NOTRANS[] = "N", ALL[] = "A", UPPER[] = "U";
    char LOWER[] = "L", LEFT[] = "L", RIGHT[] = "R", NONUNIT[] = "N";
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
    if (lwork < 5*n*s + 9*s*s + msub*n)
    {
        printf("%% Work space is too small. It should be larger than 6*n*s + 8*s*s.\n");
        return 1;
    }
    Scol = work;
    T = Scol + n*s;
    Ycol = T + s*s;
    worktemp = Ycol + 2*n*s + 3*s*s;
    Sdiag = worktemp + 2*n*s + 3*s*s;
    Ydiag = Sdiag + s*s;
    Usub = Ydiag + s*s;


    // Perform intraorthogonalization to the first block column.
    dlacpy_(ALL, &msub, &s, Xsub, &ldXsub, Usub, &msub, 1);
    TSQR::Tsqr<int, double> mytsqr = init_tsqr();
    mytsqr.factorExplicit(msub, s0, Usub, msub, Xsub, ldXsub, R, ldR, false);

    // Update Q2.
    ntemp = s+s0;
    dgemm_(TRANS, NOTRANS, &s0, &s, &msub, &one, Xsub, &ldXsub,
            &Xsub[s0*ldXsub], &ldXsub, &zero, Scol, &s0, 1, 1);
    MPI_Allreduce(Scol, worktemp, s0*s, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    dlacpy_(ALL, &s0, &s, worktemp, &s0, &R[s0*ldR], &ldR, 1);// Scol stores in R.
    dgemm_(NOTRANS, NOTRANS, &msub, &s, &s0, &rone, Xsub, &ldXsub, worktemp,
            &s0, &one, &Xsub[ldXsub*s0], &ldXsub, 1, 1);
    dlacpy_(ALL, &msub, &s, &Xsub[ldXsub*s0], &ldXsub, Usub, &msub, 1);
    mytsqr.factorExplicit(msub, s, Usub, msub, &Xsub[ldXsub*s0], ldXsub,
            Sdiag, s, false);
    nZ = MIN(s, n-s-s0);
    s2 = s + nZ;
    dgemm_(TRANS, NOTRANS, &ntemp, &s2, &msub, &one, Xsub, &ldXsub,
            &Xsub[s0*ldXsub], &ldXsub, &zero, Ycol, &ntemp, 1, 1);
    MPI_Allreduce(Ycol, worktemp, ntemp*s2, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    dgemm_(TRANS, NOTRANS, &s, &s, &s0, &rone, worktemp, &ntemp, worktemp,
            &ntemp, &one, worktemp+s0, &ntemp, 1, 1);
    dlacpy_(UPPER, &s, &s, worktemp+s0, &ntemp, Ydiag, &s, 1);
    dpotrf_(UPPER, &s, Ydiag, &s, &info, 1);
    dgemm_(NOTRANS, NOTRANS, &msub, &s, &s0, &rone, Xsub, &ldXsub,
            worktemp, &ntemp, &one, &Xsub[ldXsub*s0], &ldXsub, 1, 1);
    dtrsm_(RIGHT, UPPER, NOTRANS, NONUNIT, &msub, &s, &one, Ydiag, &s,
            &Xsub[ldXsub*s0], &ldXsub, 1, 1, 1, 1); // U2 stores in X2.
    // Update R.
    dgemm_(NOTRANS, NOTRANS, &s0, &s, &s, &one, worktemp, &ntemp, Sdiag,
            &s, &one, &R[s0*ldR], &ldR, 1, 1);
    dgemm_(NOTRANS, NOTRANS, &s, &s, &s, &one, Ydiag, &s, Sdiag, &s, &zero,
            &R[s0*ldR+s0], &ldR, 1, 1);

    mZ = s0;
    nZ = MIN(s, n-s-s0);

    for (int i = 1; i < p; i++)
    {
        mScol = mZ + s;
        dlacpy_(ALL, &mScol, &nZ, worktemp+ntemp*s, &ntemp, Scol, &mScol, 1);
        dgemm_(TRANS, NOTRANS, &s, &nZ, &mZ, &rone, worktemp, &ntemp,
                worktemp+ntemp*s, &ntemp, &one, Scol+mZ, &mScol, 1, 1);
        dtrsm_(LEFT, UPPER, TRANS, NONUNIT, &s, &nZ, &one, Ydiag, &s,
                Scol+mZ, &mScol, 1, 1, 1, 1);
        dlaset_(LOWER, &nZ, &nZ, &zero, &zero, Sdiag, &nZ, 1);
        dgemm_(NOTRANS, NOTRANS, &msub, &nZ, &mScol, &rone, Xsub, &ldXsub,
                Scol, &mScol, &one, &Xsub[ldXsub*mScol], &ldXsub, 1, 1);
        dlacpy_(ALL, &msub, &nZ, &Xsub[ldXsub*mScol], &ldXsub, Usub,
                &msub, 1);
        mytsqr.factorExplicit(msub, nZ, Usub, msub, &Xsub[ldXsub*mScol],
                ldXsub, Sdiag, nZ, false);
        if (i < p - 1)
        {
            nZ = MIN(s, n-(i+1)*s-s0);
            s2 = s + nZ;
            ntemp = mScol + s;
            dgemm_(TRANS, NOTRANS, &ntemp, &s2, &msub, &one, Xsub, &ldXsub,
                    &Xsub[mScol*ldXsub], &ldXsub, &zero, Ycol, &ntemp, 1, 1);
            dgemm_(TRANS, NOTRANS, &nZ, &nZ, &msub, &one, &Xsub[ntemp*ldXsub],
                    &ldXsub, &Xsub[ntemp*ldXsub], &ldXsub, &zero,
                    Ycol+ntemp*s2, &nZ, 1, 1);
            MPI_Allreduce(Ycol, worktemp, ntemp*s2+nZ*nZ, MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD);
            nY = s;
        }
        else
        {
            ntemp = mScol + nZ;
            dgemm_(TRANS, NOTRANS, &ntemp, &nZ, &msub, &one, Xsub, &ldXsub,
                    &Xsub[mScol*ldXsub], &ldXsub, &zero, Ycol, &ntemp, 1, 1);
            MPI_Allreduce(Ycol, worktemp, ntemp*nZ, MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD);
            nY = nZ;
        }
        dgemm_(TRANS, NOTRANS, &nY, &nY, &mScol, &rone, worktemp, &ntemp,
                worktemp, &ntemp, &one, worktemp+mScol, &ntemp, 1, 1);
        dlaset_(LOWER, &nY, &nY, &zero, &zero, Ydiag, &nY, 1);
        dlacpy_(UPPER, &nY, &nY, worktemp+mScol, &ntemp, Ydiag, &nY, 1);
        dpotrf_(UPPER, &nY, Ydiag, &nY, &info, 1);
        dgemm_(NOTRANS, NOTRANS, &msub, &nY, &mScol, &rone, Xsub, &ldXsub,
                worktemp, &ntemp, &one, &Xsub[ldXsub*mScol], &ldXsub, 1, 1);
        dtrsm_(RIGHT, UPPER, NOTRANS, NONUNIT, &msub, &nY, &one, Ydiag, &nY,
                &Xsub[ldXsub*mScol], &ldXsub, 1, 1, 1, 1);
        // Update R.
        dlacpy_(ALL, &mScol, &nY, Scol, &mScol, &R[mScol*ldR], &ldR, 1);
        dgemm_(NOTRANS, NOTRANS, &mScol, &nY, &nY, &one, worktemp, &ntemp,
                Sdiag, &nY, &one, &R[mScol*ldR], &ldR, 1, 1);
        dgemm_(NOTRANS, NOTRANS, &nY, &nY, &nY, &one, Ydiag, &nY, Sdiag,
                &nY, &zero, &R[mScol*ldR+mScol], &ldR, 1, 1);

        mZ = mZ + s;
    }
    //if (myrank_mpi == 0)
    //    printf("p=%d, mZ=%d, nZ=%d, nY=%d\n", p, mZ, nZ, nY);
    return 0;
}

