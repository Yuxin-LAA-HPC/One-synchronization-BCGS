This code is for comparing different variants of reorthogonalized block
classical Gram--Schmidt (BCGS) algorithms on distributed memory systems, according
to our paper:

    A stable one-synchronization variant of reorthogonalized block classical Gram--Schmidt.
    Erin Carson and Yuxin Ma. (2024) arXiv:2411.07077

The following variants have been implementated in this code:

BCGSI+ (function bcgsi2): Also named as BCGS2, the classical reorthogonalized BCGS algorithm;

BCGS-PIPI+ (function bcgspipi2): A two-sync variant proposed by E. Carson, K. Lund, Y. Ma, and E. Oktay: Reorthogonalized Pythagorean variants of block classical Gram--Schmidt. SIAM Journal on Matrix Analysis and Applications 46(1), 310--340 (2025).

BCGSI+P-1S (function bcgsi2P1s): Our new one-sync variant.

BCGSI+P-2S (function bcgsi2P2s): Our new two-sync variant.

BCGSI+A-1S (function bcgsi21s): A one-sync variant proposed by I. Yamazaki,
S. Thomas, M. Hoemmen, E. G. Boman, K. \'Swirydowicz, and J. J. Eilliot:
Low-synchronization orthogonalization schemes for s-step and pipelined Krylov solvers in
Trilinos, in Proceedings of the 2020 SIAM Conference on Parallel Processing for Scientific
Computing (PP), pp. 118–12 (2020).

Remarks for compiling this code:

This code requires packages including

1. OPENBLAS, LAPACK, and SCALAPACK.

2. Trilinos (or the sub-package Tpetra in Trilinos).
The link of Trilinos: https://trilinos.github.io

We give an example of make.inc called make.inc.example.
The users could accordingly modify the links of these related packages to compile this code.


SRC/:

util.c: Includes functions to test the accuracy and to print matrices.

orth.c: Includes all different variants of reorthogonalized BCGS.


TESTS/:

test_orth.c: Compare the performance of different variants of reorthogonalized BCGS and also check the accuracy.
