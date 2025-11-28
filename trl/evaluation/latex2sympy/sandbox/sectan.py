from sympy import *
import sys
sys.path.append("..")

# # x^2\cdot \left(3\cdot \tan \left([!a!]\cdot x+[!c!]\right)+[!a!]\cdot x\left(\sec \left([!a!]\cdot x+[!c!]\right)\right)^2\right)
# latex1 = "x^2\\cdot \\left(3\\cdot \\tan \\left(2\\cdot x+5\\right)+2\\cdot x\\left(\\sec \\left(2\\cdot x+5\\right)\\right)^2\\right)"
# math1 = process_sympy(latex1)
# print("latex: %s to math: %s" %(latex1,math1))
#
# latex2 = "x^2\\cdot \\left(3\\cdot \\tan \\left(2\\cdot x+5\\right)+2\\cdot x\\left(\\sec \\left(2\\cdot x+5\\right)^2\\right)\\right)"
# math2 = process_sympy(latex2)
# print("latex: %s to math: %s" %(latex2,math2))
#
# latex3 = "x^2\\cdot \\left(3\\cdot \\tan \\left(2\\cdot x+5\\right)+2\\cdot x\\left(1+\\tan \\left(2\\cdot x+5\\right)^2\\right)\\right)"
# math3 = process_sympy(latex3)
# print("latex: %s to math: %s" %(latex3,math3))
#
# print(simplify(math1 - math2))
# print(simplify(math1 - math3))

#
# latex1 = "\\sec^2(2\\cdot x+5)"
# math1 = process_sympy(latex1)
# print("latex: %s to math: %s" %(latex1,math1))
#
# latex2 = "1+\\tan^2(2\\cdot x+5)"
# math2 = process_sympy(latex2)
# print("latex: %s to math: %s" %(latex2,math2))
# print(simplify(math1 - math2))


x = Symbol('x', real=True)
y = Symbol('y', real=True)

# BUG: 1 + tan^2(x+1) should be == sec^2(x+1) but isnt
lhs = (1 + (tan(x + 1))**2)
rhs = (sec(x + 1))**2
eq = lhs - rhs
print(simplify(lhs))
print(simplify(rhs))
print(simplify(eq))
print(simplify(lhs) == simplify(rhs))

# 1 + tan^2(x) == sec^2(x) but isnt
lhs = (1 + (tan(x))**2)
rhs = (sec(x))**2
eq = lhs - rhs
print(simplify(lhs))
print(simplify(rhs))
print(simplify(eq))
print(simplify(lhs) == simplify(rhs))
