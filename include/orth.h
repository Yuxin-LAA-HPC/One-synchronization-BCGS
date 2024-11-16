#ifndef ORTH_H
#define ORTH_H


#ifdef __cplusplus
extern "C" {
#endif
int bcgsi2P1s(int m, int n, int s, double *Xsub, int ldXsub, double *R,
        int ldR, double *work, int lwork);
int bcgsi21s(int m, int n, int s, double *Xsub, int ldXsub, double *R,
        int ldR, double *work, int lwork);
int bcgspipi2(int m, int n, int s, double *Xsub, int ldXsub, double *R,
        int ldR, double *work, int lwork);
int bcgsi2(int m, int n, int s, double *Xsub, int ldXsub, double *R,
        int ldR, double *work, int lwork);

//TSQR::Tsqr<int, double> init_tsqr();
//int intraorth_tsqr(int msub, int s0, double *Xsub, int ldXsub, double *Qsub,
//        int ldQsub, double *R, int ldR);
#ifdef __cplusplus
}
#endif

#endif
