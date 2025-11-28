from latex2sympy import process_sympy
from sympy import *
import sys
sys.path.append("..")

theta = Symbol('theta', real=True)

latex = "\\begin{matrix}1&2\\\\3&4\\end{matrix}"
math = process_sympy(latex)
print("latex: %s to math: %s" % (latex, math))

latex = "\\begin{matrix}1&2\\\\3&4\\\\5&6\\end{matrix}"
math = process_sympy(latex)
print("latex: %s to math: %s" % (latex, math))

latex = "\\begin{matrix}1&2&3\\\\4&5&6\\\\7&8&9\\end{matrix}"
math = process_sympy(latex)
print("latex: %s to math: %s" % (latex, math))

latex = "\\begin{matrix}x^1&x^2&x^3\\\\y^1&y^2&y^3\\\\z^1&z^2&z^3\\end{matrix}"
math = process_sympy(latex)
print("latex: %s to math: %s" % (latex, math))

latex = "\\begin{matrix}x\\\\y\\end{matrix}"
math = process_sympy(latex)
print("latex: %s to math: %s" % (latex, math))

latex = "2\\cdot\\begin{matrix}x\\\\y\\end{matrix}"
math = process_sympy(latex)
print("latex: %s to math: %s" % (latex, math))

latex = "2\\cdot\\begin{matrix}x\\\\y\\end{matrix} + \\begin{matrix}2\\\\3\\end{matrix}"
math = process_sympy(latex)
print("latex: %s to math: %s" % (latex, math))

latex = "-2\\begin{matrix}1&2\\\\3&4\\end{matrix}"
math = process_sympy(latex)
print("latex: %s to math: %s" % (latex, math))

latex = "2\\cdot\\theta\\begin{matrix}x\\\\y\\end{matrix} + \\begin{matrix}2\\\\3\\end{matrix}"
math = process_sympy(latex)
print("latex: %s to math: %s" % (latex, math))

latex = "\\theta\\begin{matrix}1\\\\3\\end{matrix} - \\begin{matrix}-1\\\\2\\end{matrix}"
math = process_sympy(latex)
print("latex: %s to math: %s" % (latex, math))
