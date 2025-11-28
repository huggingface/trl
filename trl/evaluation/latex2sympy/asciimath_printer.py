from sympy.printing.str import StrPrinter
from sympy.core import S

class AsciiMathPrinter(StrPrinter):

    def _print_Limit(self, expr):
        e, z = expr.args

        return "lim_(%s -> %s) %s" % (self._print(z), self._print(z), self._print(e))

    def _print_Integral(self, expr):
        e, lims = expr.args
        if len(lims) > 1:
            return "int_(%s)^(%s) %s d%s" % (self._print(lims[1]), self._print(lims[2]), self._print(e), self._print(lims[0]))
        else:
            return "int %s d%s" % (self._print(e), self._print(lims))
    
    def _print_Sum(self, expr):
        e, lims = expr.args
        return "sum_(%s = %s)^(%s) %s" % (self._print(lims[0]), self._print(lims[1]), self._print(lims[2]), self._print(e))

    def _print_Product(self, expr):
        e, lims = expr.args
        return "prod_(%s = %s)^(%s) %s" % (self._print(lims[0]), self._print(lims[1]), self._print(lims[2]), self._print(e))

    def _print_factorial(self, expr):
        return "%s!" % self._print(expr.args[0])

    def _print_Derivative(self, expr):
        e = expr.args[0]
        wrt = expr.args[1]
        return "d/d%s %s" % (self._print(wrt), self._print(e))

    def _print_Abs(self, expr):
        return "|%s|" % self._print(expr.args[0])

    def _print_Equality(self, expr):
        return "%s = %s" % (self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_Pow(self, expr):
        b = self._print(expr.base)
        if expr.exp is S.Half:
            return "sqrt(%s)" % b

        if -expr.exp is S.Half:
            return "1/sqrt(%s)" % b
        if expr.exp is -S.One:
            return "1/%s" % b

        return "%s^(%s)" % (b, self._print(expr.exp)) 
