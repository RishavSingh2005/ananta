import math
import re
import statistics
import ast

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from .tools import ToolError


_SAFE_EXPRESSION = re.compile(r"^[0-9A-Za-z_+\-*/^().,=\[\]{}\s]+$")
_SAFE_NAMES = {
    "E": sp.E,
    "I": sp.I,
    "N": sp.N,
    " pi": sp.pi,
    "pi": sp.pi,
    "oo": sp.oo,
    "sqrt": sp.sqrt,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "log": sp.log,
    "exp": sp.exp,
    "Abs": sp.Abs,
}


def _expression(value):
    if not value or len(value) > 500:
        raise ToolError("mathematical expression must be 1-500 characters")
    if "__" in value or not _SAFE_EXPRESSION.fullmatch(value):
        raise ToolError("expression contains unsupported characters")
    try:
        return parse_expr(value, local_dict=_SAFE_NAMES, evaluate=True)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ToolError("could not parse mathematical expression") from error


def symbolic_math(operation, expression, variable="x"):
    operation = operation.casefold().strip()
    if operation == "statistics":
        try:
            values = [float(item.strip()) for item in expression.split(",")]
            if not values:
                raise ValueError
            result = {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values), "min": min(values), "max": max(values)}
            if len(values) > 1:
                result["stdev"] = statistics.stdev(values)
            return result
        except (TypeError, ValueError, statistics.StatisticsError) as error:
            raise ToolError("statistics expects comma-separated numbers") from error

    if operation == "matrix":
        if not expression.startswith("[") or not expression.endswith("]"):
            raise ToolError("matrix expects nested lists, for example [[1,2],[3,4]]")
        try:
            matrix = sp.Matrix(ast.literal_eval(expression))
        except Exception as error:
            raise ToolError("invalid matrix") from error
        return {"matrix": str(matrix), "determinant": str(matrix.det()) if matrix.rows == matrix.cols else None, "rank": matrix.rank(), "shape": [matrix.rows, matrix.cols]}

    parsed = _expression(expression)
    symbol = sp.Symbol(variable)
    try:
        if operation == "simplify":
            result = sp.simplify(parsed)
        elif operation == "solve":
            result = sp.solve(parsed, symbol)
        elif operation == "differentiate":
            result = sp.diff(parsed, symbol)
        elif operation == "integrate":
            result = sp.integrate(parsed, symbol)
        elif operation == "limit":
            result = sp.limit(parsed, symbol, 0)
        elif operation == "evaluate":
            result = sp.N(parsed)
        elif operation == "expand":
            result = sp.expand(parsed)
        elif operation == "factor":
            result = sp.factor(parsed)
        elif operation == "plot":
            points = []
            for index in range(-10, 11):
                value = parsed.subs(symbol, index)
                if value.is_real and value.is_finite:
                    points.append({"x": index, "y": float(value)})
            return {"expression": str(parsed), "variable": variable, "points": points}
        else:
            raise ToolError("unsupported math operation")
    except (ValueError, TypeError, ZeroDivisionError) as error:
        raise ToolError("mathematical operation failed") from error
    return {"operation": operation, "expression": str(parsed), "result": str(result)}