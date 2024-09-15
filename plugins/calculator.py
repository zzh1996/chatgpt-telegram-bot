import ast
import operator
import math

ops = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.BitAnd: operator.and_,
    ast.FloorDiv: operator.floordiv,
    ast.Invert: operator.invert,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

consts = {
    'pi': math.pi,
    'e': math.e,
}

funcs = {
    'sqrt': math.sqrt,
    'exp': math.exp,
    'log': math.log,
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'asin': math.asin,
    'acos': math.acos,
    'atan': math.atan,
}

def evaluate(node):
    if isinstance(node, ast.Expression):
        return evaluate(node.body)
    elif isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, int) or isinstance(value, float):
            return value
        else:
            raise ValueError(f"Constant of type {type(value)} not supported")
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in ops:
            raise ValueError(f"Operator {type(node.op)} not supported")
        left = evaluate(node.left)
        right = evaluate(node.right)
        if type(node.op) == ast.Pow:
            if isinstance(left, int) and isinstance(right, int) and abs(left) > 1 and abs(right) > 1:
                if left.bit_length() * right > 1000:
                    right = float(right)
        if type(node.op) == ast.LShift:
            if isinstance(left, int) and isinstance(right, int) and left != 0:
                if left.bit_length() + right > 1000:
                    raise ValueError(f"LShift operand too large")
        return ops[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in ops:
            raise ValueError(f"Operator {type(node.op)} not supported")
        return ops[type(node.op)](evaluate(node.operand))
    elif isinstance(node, ast.Name):
        if node.id not in consts:
            raise ValueError(f"Constant {type(node)} not supported")
        return consts[node.id]
    elif isinstance(node, ast.Call):
        assert isinstance(node.func, ast.Name)
        if node.func.id not in funcs:
            raise ValueError(f"Function {node.func.id} not supported")
        func = funcs[node.func.id]
        return func(*(evaluate(node) for node in node.args))

class Calculator:
    functions = [{
        "name": "calc",
        "description": "Evaluate Python-style math expression (but not real Python). Allowed operators: +, -, *, /, //, %, **, <<, >>, |, &, ^, ~, pi, e, sqrt, exp, log, sin, cos, tan, asin, acos, atan. Functions are available directly, don't import math.",
        "parameters": {
            "type": "object",
            "properties": {
                "expr": {
                    "type": "string",
                    "description": "Python-style math expression",
                },
            },
            "required": ["expr"],
            "additionalProperties": False,
        },
        "strict": True,
    }]

    def calc(self, expr):
        expr = expr.replace('math.', '')
        root = ast.parse(expr, mode='eval')
        return str(evaluate(root))

if __name__ == '__main__':
    c = Calculator()
    assert c.calc("1+1") == 2
    print(c.calc("pi + 1e-3 * sin(-123)"))
