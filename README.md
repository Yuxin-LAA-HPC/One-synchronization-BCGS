This code is for comparing different low-synchronization variants of reorthogonalized block
classical Gram--Schmidt (BCGS) algorithms on distributed memory systems, according
to our paper:

    A stable one-synchronization variant of reorthogonalized block classical Gram--Schmidt.
    Erin Carson and Yuxin Ma. (2024) arXiv:2411.07077

The following variants have been implementated in this code:

* BCGSI+ (function bcgsi2): Also known as BCGS2, the classical reorthogonalized BCGS algorithm.

* BCGS-PIPI+ (function bcgspipi2): A two-synchronization variant proposed by [1].

* BCGSI+P-1S (function bcgsi2P1s): Our new one-synchronization variant.

* BCGSI+P-2S (function bcgsi2P2s): Our new two-synchronization variant.

* BCGSI+A-1S (function bcgsi21s): A one-synchronization variant proposed by [2].

## Remarks for running this code

### Envionment

This code requires packages including

* OPENBLAS, LAPACK, and SCALAPACK.

* Trilinos (or the sub-package Tpetra in Trilinos).
The link of Trilinos: https://trilinos.github.io

We give an example of make.inc called make.inc.example.
The users could accordingly modify the links of these related packages to compile this code.

### Structure of the code

* SRC/:

+ util.c: Includes functions to test the accuracy and to print matrices.

+ orth.c: Includes all different variants of reorthogonalized BCGS.


* TESTS/:

+ test_orth.c: Compare the performance of different variants of reorthogonalized BCGS and also check the accuracy.

## Acknowledgement

This project is supported by the European Union (ERC, InEXASCALE, 101075632).
Views and opinions expressed are those of the authors only and do not necessarily reflect those of the European Union or the European Research Council.
Neither the European Union nor the granting authority can be held responsible for them.

## References
[1] E. Carson, K. Lund, Y. Ma, and E. Oktay. Reorthogonalized Pythagorean variants of block classical Gram--Schmidt. SIAM Journal on Matrix Analysis and Applications 46(1), 310--340 (2025).

[2] I. Yamazaki, S. Thomas, M. Hoemmen, E. G. Boman, K. Swirydowicz, and J. J. Eilliot.
Low-synchronization orthogonalization schemes for s-step and pipelined Krylov solvers in
Trilinos, in Proceedings of the 2020 SIAM Conference on Parallel Processing for Scientific
Computing (PP), pp. 118–12 (2020).
