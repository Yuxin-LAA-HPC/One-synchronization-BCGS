#ifndef SCALAPACK_H
#define SCALAPACK_H


#define MIN(x, y) ((x) <= (y) ? (x) : (y))
#define MAX(x, y) ((x) >= (y) ? (x) : (y))

#ifdef __cplusplus
extern "C" {
#endif
    /* Cblacs declarations */
    void Cblacs_pinfo(int*, int*);
    void Cblacs_get(int, int, int*);
    void Cblacs_gridinit(int*, const char*, int, int);
    void Cblacs_pcoord(int, int, int*, int*);
    void Cblacs_gridexit(int);
    void Cblacs_barrier(int, const char*);
    void Cdgerv2d(int, int, int, double*, int, int, int);
    void Cdgesd2d(int, int, int, double*, int, int, int);
    void Cblacs_gridinfo( int, int*, int*, int*, int* );
    
    /* ScaLAPACK declarations */
    int  numroc_(int*, int*, int*, int*, int*);
    void descinit_( int *desc, int *m, int *n, int *mb, int *nb, int *irsrc,
            int *icsrc, int *ictxt, int *lld, int *info);
    void pdgemm_(char *transa, char *transb, int *M, int *N, int *K, double *alpha,
            double *A, int *ia, int *ja, int *desca,
            double *B, int *ib, int *jb, int *descb, double *beta,
            double *C, int *ic, int *jc, int *descc);
    void pdnrm2_(int *N, double *norm2, double *X, int *ix, int *jx,
            int *descx, int *INCX);
    void pdgeqrf_(int *M, int *N, double *A, int *IA, int *JA, int *DESCA,
            double *TAU, double *WORK, int *LWORK, int *INFO);
    void pdormqr_(char *SIDE, char *TRANS, int *M, int *N, int *K, double *A,
            int *IA, int *JA, int *DESCA, double *TAU, double *C, int *IC,
            int *JC, int *DESCC, double *WORK, int *LWORK, int *INFO);
#ifdef __cplusplus
}
#endif


#endif
