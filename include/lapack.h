#ifndef LAPACK_H
#define LAPACK_H


#define MIN(x, y) ((x) <= (y) ? (x) : (y))
#define MAX(x, y) ((x) >= (y) ? (x) : (y))

#ifdef __cplusplus
extern "C" {
#endif
    // BLAS 1.
    void dcopy_(int *N, double *DX, int *INCX, double *DY, int *INCY);
    void dscal_(int *N, double *DA, double *DX, int *INCX);
    double dnrm2_(int *N, double *DX, int *INCX);

    // BLAS 2.
    void dlaset_(char *UPLO, int *M, int *N, double *ALPHA, double *BETA,
            double *A, int *LDA, int);
    void dlacpy_(char *UPLO, int *M, int *N, double *A, int *LDA, double *B,
            int *LDB, int);

    // BLAS 3.
    void dgemm_(char *TRANSA, char *TRANSB, int *M, int *N, int *K,
            double *ALPHA, double A[], int *LDA, double B[], int *LDB,
            double *BETA, double C[], int *LDC, int, int);
    void dtrsm_(char *SIDE, char *UPLO, char *TRANSA, char *DIAG, int *M,
            int *N, double *ALPHA, double A[], int *LDA, double B[],
            int *LDB, int, int, int, int);
    void dpotrf_(char *UPLC, int *N, double A[], int *LDA, int *INFO, int);
#ifdef __cplusplus
}
#endif


#endif
