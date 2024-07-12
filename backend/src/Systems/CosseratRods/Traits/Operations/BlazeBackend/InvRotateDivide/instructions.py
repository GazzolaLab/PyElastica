import sympy as sp
from sympy.utilities.codegen import codegen

Q_curr = sp.MatrixSymbol("Q_curr", 3, 3)
Q_next = sp.MatrixSymbol("Q_next", 3, 3)
R = Q_curr * sp.Transpose(Q_next)

trR = sp.Trace(R).rewrite(sp.Sum)

arg = R - sp.Transpose(R)


[(c_name, c_code), (h_name, c_header)] = codegen(
    ("f", trR.simplify()), "C89", "test", header=False, empty=False
)
print(c_code)


[(c_name, c_code), (h_name, c_header)] = codegen(
    ("f", arg.simplify()), "C89", "test", header=False, empty=False
)
print(c_code)
