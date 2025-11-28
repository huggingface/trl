from latex2sympy import process_sympy
from sympy import *
import sys
import hashlib
import time

sys.path.append("..")


M = Matrix([[1, 2], [3, 4]])
v = Matrix([1, 2])

# sub settings
sub_settings_symbols = {}
sub_settings_symbols[Symbol('M' + hashlib.md5('M'.encode()).hexdigest(), real=True)] = M
sub_settings_symbols[Symbol('v' + hashlib.md5('v'.encode()).hexdigest(), real=True)] = v


# one parameters
latex = "\\begin{matrix}1&2\\\\3&4\\end{matrix}\\cdot[!v!]"
equation_sympy_check = MatMul(M, Symbol('v' + hashlib.md5('v'.encode()).hexdigest(), real=True))
equation_sympy_subs_check = MatMul(M, v)
# placeholders
equation_sympy = process_sympy(latex)
print('latex = %s' % latex)
print('equation_sympy = %s' % equation_sympy)
print('equation_sympy_check = %s' % equation_sympy_check)
print('equation_sympy = %s' % (srepr(equation_sympy)))

equation_sympy_subs = equation_sympy.subs(sub_settings_symbols, evaluate=False)
print('equation_sympy_subs = %s' % equation_sympy_subs)
print('equation_sympy_subs_check = %s' % equation_sympy_subs_check)


# two parameters

# sub settings
print('')
print('============== Two Parameters -> M*v = Matrix*Vector =============')
sub_settings_symbols = {}
sub_settings_symbols[Symbol('M' + hashlib.md5('M'.encode()).hexdigest(), commutative=False)] = M
sub_settings_symbols[Symbol('v' + hashlib.md5('v'.encode()).hexdigest(), commutative=False)] = v

latex = "[!M!]\\cdot[!v!]"
math_check = Mul(Symbol('M' + hashlib.md5('M'.encode()).hexdigest(), commutative=False), Symbol('v' + hashlib.md5('v'.encode()).hexdigest(), commutative=False))
# placeholders
equation_sympy = process_sympy(latex)
print(latex)
print(math_check)
print(equation_sympy)
print(srepr(equation_sympy))

# performance
t0 = time.time()

# process_sympy and substitute at the same time
# Only needed for linalg input
placeholder_values = {'M': M, 'v': v}
equation_sympy_subs = process_sympy(latex, variable_values=placeholder_values)

t1 = time.time()
print('equation with substituted placeholders = %s' % (str(equation_sympy_subs)))
print('time to process to sympy with placeholders = %s s' % (t1 - t0))
print('')
print('============== Two Parameters -> M*v = Matrix*Vector =============')
