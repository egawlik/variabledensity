# variabledensity
### A conservative finite element method for the incompressible Euler equations with variable density

This code implements the numerical method described in

E. S. Gawlik & F. Gay-Balmaz. A Conservative Finite Element Method for the Incompressible Euler Equations with Variable Density. Journal of Computational Physics, 412, 109439 (2020). [ [pdf](http://math.hawaii.edu/~egawlik/pdf/GaGB2019b.pdf) | [doi](https://doi.org/10.1016/j.jcp.2020.109439) ]

To run it, install [FEniCS](https://fenicsproject.org) and type, e.g.,

python3 variabledensity.py -o 0 -d 1 -t 5.0 -k 0.01 -a 0.5 -b 0.5 -r 3

The optional command line arguments above are explained in the code.

Acknowledgements: This work was supported in part by an NSF grant (DMS-1703719) and an ANR grant (GEOMFLUID, ANR-14-CE23-0002-01).  When writing this code, I used a .py file written by Maurizio Chiaramonte as a template.
