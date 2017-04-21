This is code used in the numerical experiments of the paper

"Outer-Product-Free Sets for Polynomial Optimization and Oracle-Based Cuts"

by Bienstock D. Chen C. and Muñoz G. Available on arXiv 1610.04604.

The code uses as input any model readable by Gurobi (LP, MPS, REW, etc), and then executes
a cutting plane procedure used for finding intersection cuts as indicated in the paper. 
The input model must be a quadratic problem, although not necessarily convex.

We include 2 folders with the test instances used in the paper in LP format, as well as the
full table of results obtained.

To run the code, execute

python intersectioncuts.py filename.lp

You can also include an additional file to indicate a special set of cut families to be considered.
The file "optMinorPSD.txt" is included as an example, and in such case the code must be executed as

python intersectioncuts.py filename.lp optMinorPSD.txt

The code requires Numpy, Scipy, and Gurobi installed.

This code is not intended as a serious implementation, nor a standalone solver, but rather 
as a tool for testing these new families of cuts. Some results may vary in different machines
due to numerical instability of some instances.

Any feedback is appreciated, and please cite if used.
