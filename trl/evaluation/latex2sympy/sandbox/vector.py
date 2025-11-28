import numpy as np
from sympy import *
import sys
sys.path.append("..")

# row column matrix = vector
v = [1, 2, 3]

# single column matrix = vector
m = Matrix([1, 2, 3])
print(m[:, 0])

# a three row and 2 column matrix
m = Matrix([[1, 2], [3, 4], [5, 6]])
print(m[:, 0])

# determinant of lin indp system != 0
m = Matrix([[1, 1], [1, 2]])
print(m.det())

# determinant of lin dep system = 0
m = Matrix([[1, 1], [2, 2]])
print(m.det())

# determinant of lin dep system = 0
x = Symbol('x')
y = Symbol('y')
m = Matrix([[x, y], [x, y]])
print(m.det())
# Reduced Row-Echelon Form
_, ind = m.rref()
print(len(ind))

# determinant of lin dep system != 0
m = Matrix([[x, y], [y, x]])
print(m.det())
# Reduced Row-Echelon Form
_, ind = m.rref()
print(len(ind))

# determinant of lin dep system != 0
# Reduced Row-Echelon Form
m = Matrix([[x, x, y], [y, y, y]])
_, ind = m.rref()
# Reduced Row-Echelon Form
print(len(ind))

#==================#
#===== Numpy ======#
#==================#
# http://kitchingroup.cheme.cmu.edu/blog/2013/03/01/Determining-linear-independence-of-a-set-of-vectors/
# Lin Indp of set of numerical vectors
TOLERANCE = 1e-14
v1 = [6, 0, 3, 1, 4, 2]
v2 = [0, -1, 2, 7, 0, 5]
v3 = [12, 3, 0, -19, 8, -11]

A = np.row_stack([v1, v2, v3])

U, s, V = np.linalg.svd(A)
print(s)
print(np.sum(s > TOLERANCE))

v1 = [1, 1]
v2 = [4, 4]

A = np.row_stack([v1, v2])
U, s, V = np.linalg.svd(A)
print(s)
print(np.sum(s > TOLERANCE))


latex = "\\begin{matrix}1&2\\\\3&4\\end{matrix}"
# math = process_sympy(latex)
print("latex: %s to math: %s" % (latex, 1))
