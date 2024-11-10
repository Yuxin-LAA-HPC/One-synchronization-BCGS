#ifndef UTIL_H
#define UTIL_H


#ifdef __cplusplus
extern "C" {
#endif
    void dprintmat(char *name, int nrows, int ncols, double A[], int ldA);
    void dprintmat_mpi(char *name, int nrows, int ncols, int nrowstart,
            int nrowssub, int ncolssub, double Asub[], int ldAsub);

    int test_orth_accuracy(int m, int n, int msub, double *Xsub, int ldXsub,
            double *Qsub, int ldQsub, double *R, int ldR,
            double *work, int lwork);
#ifdef __cplusplus
}
#endif


#endif
