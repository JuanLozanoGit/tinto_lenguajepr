import sys
import os
from antlr4 import *

sys.path.append(os.path.join(os.path.dirname(__file__), 'antlr_gen', 'grammar'))
from TintoLexer import TintoLexer
from TintoParser import TintoParser
from TintoVisitor import TintoVisitor

class ReturnException(Exception):
    def __init__(self, value):
        self.value = value

# Pila implementada manualmente
class Stack:
    def __init__(self):
        self._data = []
    def push(self, v): self._data.append(v)
    def pop(self):
        if not self._data: raise Exception("pop de pila vacía")
        return self._data.pop()
    def top(self):
        if not self._data: raise Exception("top de pila vacía")
        return self._data[-1]
    def is_empty(self): return len(self._data) == 0
    def size(self): return len(self._data)
    def __repr__(self): return f"stack({self._data})"

class TintoInterpreter(TintoVisitor):
    def __init__(self):
        super().__init__()
        self.globals = {
            "PI": 3.141592653589793, "E": 2.718281828459045,
            "stack": lambda: Stack(),
            "push": lambda s, v: s.push(v),
            "pop": lambda s: s.pop(),
            "top": lambda s: s.top(),
            "isEmpty": lambda s: s.is_empty(),
            "size": lambda s: s.size()
        }
        self.scopes = [self.globals]
        self.functions = {}
        self.memo = {}
        self.max_depth = 1000

    def current_scope(self): return self.scopes[-1]

    # ---------- Funciones matemáticas manuales ----------
    def factorial(self, n):
        r = 1
        for i in range(1, int(n)+1): r *= i
        return r

    def seno(self, grados):
        x = grados * self.globals["PI"] / 180
        res = 0
        for n in range(10):
            signo = -1 if n%2 else 1
            res += signo * (x**(2*n+1)) / self.factorial(2*n+1)
        return res

    def coseno(self, grados):
        x = grados * self.globals["PI"] / 180
        res = 0
        for n in range(10):
            signo = -1 if n%2 else 1
            res += signo * (x**(2*n)) / self.factorial(2*n)
        return res

    def tangente(self, grados):
        c = self.coseno(grados)
        if c == 0: raise Exception("Tangente indefinida")
        return self.seno(grados) / c

    def raiz(self, n):
        if n < 0: raise Exception("Raíz de negativo")
        if n == 0: return 0
        x = n
        for _ in range(20): x = (x + n/x) / 2
        return x

    def logaritmo(self, n):
        if n <= 0: raise Exception("Logaritmo inválido")
        x = 1
        for _ in range(20):
            ex = self.globals["E"] ** x
            x = x - (ex - n) / ex
        return x

    # ---------- Evaluar una expresión a partir de un texto ----------
    def _eval_expr(self, text):
        lexer = TintoLexer(InputStream(text))
        parser = TintoParser(CommonTokenStream(lexer))
        return self.visit(parser.expr())

    # ---------- Parsear literales de listas, diccionarios y matrices ----------
    def _split_outer(self, text, sep, start, end):
        """Divide el texto en niveles superiores (ignorando anidación)"""
        parts = []
        cur = ""
        nivel = 0
        dentro_string = False
        for ch in text:
            if ch == '"' and (len(cur)==0 or cur[-1]!='\\'):
                dentro_string = not dentro_string
            if ch in '{[' and not dentro_string: nivel += 1
            if ch in '}]' and not dentro_string: nivel -= 1
            if ch == sep and nivel == 0 and not dentro_string:
                parts.append(cur.strip())
                cur = ""
            else:
                cur += ch
        if cur.strip(): parts.append(cur.strip())
        return parts

    def visitListaLiteral(self, ctx):
        inner = ctx.getText()[1:-1].strip()
        if not inner: return []
        return [self._eval_expr(elem) for elem in self._split_outer(inner, ',', '[', ']')]

    def visitDiccionarioLiteral(self, ctx):
        inner = ctx.getText()[1:-1].strip()
        if not inner: return {}
        d = {}
        for par in self._split_outer(inner, ',', '{', '}'):
            if ':' not in par: raise Exception(f"Par inválido: {par}")
            k, v = par.split(':', 1)
            d[self._eval_expr(k)] = self._eval_expr(v)
        return d

    def visitMatriz(self, ctx):
        inner = ctx.getText()[1:-1].strip()
        if not inner: return []
        matriz = []
        for fila in self._split_outer(inner, ';', '[', ']'):
            matriz.append([self._eval_expr(elem) for elem in self._split_outer(fila, ',', '[', ']')])
        return matriz

    def visitStackCreation(self, ctx): return Stack()

    # ---------- Estructura base ----------
    def visitProgram(self, ctx):
        for stmt in ctx.statement(): self.visit(stmt)
    def visitBloque(self, ctx):
        for stmt in ctx.statement(): self.visit(stmt)

    # ---------- Variables ----------
    def visitVariableDeclaration(self, ctx):
        name = ctx.ID().getText()
        value = self.visit(ctx.expr())
        self.current_scope()[name] = value
        return value

    def visitAssignment(self, ctx):
        name = ctx.ID().getText()
        value = self.visit(ctx.expr())
        for scope in reversed(self.scopes):
            if name in scope:
                scope[name] = value
                return value
        self.current_scope()[name] = value
        return value

    def visitAssignmentIndexed(self, ctx):
        name = ctx.ID().getText()
        obj = None
        for scope in reversed(self.scopes):
            if name in scope:
                obj = scope[name]
                break
        if obj is None: raise NameError(f"Variable '{name}' no definida")
        index = self.visit(ctx.expr(0))
        value = self.visit(ctx.expr(1))
        if isinstance(index, float) and index.is_integer():
            index = int(index)
        if isinstance(obj, list):
            if not isinstance(index, int): raise TypeError("Índice debe ser entero")
            while len(obj) <= index: obj.append(None)
            obj[index] = value
        elif isinstance(obj, dict):
            obj[index] = value
        else:
            raise TypeError(f"Tipo {type(obj)} no soporta indexación")
        return value

    def visitVariable(self, ctx):
        name = ctx.ID().getText()
        for scope in reversed(self.scopes):
            if name in scope: return scope[name]
        raise NameError(f"Variable '{name}' no definida")

    # ---------- Control de flujo ----------
    def visitIfStatement(self, ctx):
        if self.visit(ctx.expr()):
            self.visit(ctx.bloque(0))
        elif ctx.bloque(1):
            self.visit(ctx.bloque(1))

    def visitWhileStatement(self, ctx):
        while self.visit(ctx.expr()):
            self.visit(ctx.bloque())

    def visitForStatement(self, ctx):
        var = ctx.ID().getText()
        start = int(self.visit(ctx.expr(0)))
        end = int(self.visit(ctx.expr(1)))
        for i in range(start, end+1):
            self.current_scope()[var] = i
            self.visit(ctx.bloque())

    def visitReturnStatement(self, ctx):
        raise ReturnException(self.visit(ctx.expr()))

    # ---------- Operadores aritméticos y lógicos ----------
    def visitPotencia(self, ctx): return self.visit(ctx.expr(0)) ** self.visit(ctx.expr(1))
    def visitMulDivMod(self, ctx):
        l, r = self.visit(ctx.expr(0)), self.visit(ctx.expr(1))
        op = ctx.op.text
        if op == '*': return l * r
        if op == '/': return l / r
        return l % r
    def visitSumaResta(self, ctx):
        l, r = self.visit(ctx.expr(0)), self.visit(ctx.expr(1))
        return l + r if ctx.op.text == '+' else l - r
    def visitComparacion(self, ctx):
        l, r = self.visit(ctx.expr(0)), self.visit(ctx.expr(1))
        op = ctx.op.text
        ops = {'==': lambda a,b: a==b, '!=': lambda a,b: a!=b, '<': lambda a,b: a<b,
               '>': lambda a,b: a>b, '<=': lambda a,b: a<=b, '>=': lambda a,b: a>=b}
        return ops[op](l, r)
    def visitParentesis(self, ctx): return self.visit(ctx.expr())
    def visitNot(self, ctx): return not self.visit(ctx.expr())
    def visitAnd(self, ctx): return self.visit(ctx.expr(0)) and self.visit(ctx.expr(1))
    def visitOr(self, ctx): return self.visit(ctx.expr(0)) or self.visit(ctx.expr(1))

    # ---------- Indexación ----------
    def visitIndexacion(self, ctx):
        obj = self.visit(ctx.expr(0))
        index = self.visit(ctx.expr(1))
        if isinstance(index, float) and index.is_integer(): index = int(index)
        if isinstance(obj, list):
            if not isinstance(index, int): raise TypeError("Índice debe ser entero")
            if index < 0 or index >= len(obj): raise IndexError(f"Índice {index} fuera de rango")
            return obj[index]
        elif isinstance(obj, dict):
            if index not in obj: raise KeyError(f"Clave '{index}' no encontrada")
            return obj[index]
        else:
            raise TypeError(f"Tipo {type(obj)} no soporta indexación")

    # ---------- Funciones matemáticas (sin math) ----------
    def visitFuncMat(self, ctx):
        nombre = ctx.getChild(0).getText()
        valor = self.visit(ctx.expr())
        if nombre == "sin": return self.seno(valor)
        if nombre == "cos": return self.coseno(valor)
        if nombre == "tan": return self.tangente(valor)
        if nombre == "sqrt": return self.raiz(valor)
        if nombre == "log": return self.logaritmo(valor)
        raise Exception("Función matemática no válida")

    # ---------- Funciones definidas por el usuario ----------
    def visitFunctionDeclaration(self, ctx):
        name = ctx.ID().getText()
        params = [p.getText() for p in ctx.parameters().ID()] if ctx.parameters() else []
        self.functions[name] = {"params": params, "body": ctx.bloque()}

    def visitFunctionCall(self, ctx):
        name = ctx.ID().getText()
        if name in self.globals and callable(self.globals[name]):
            return self.globals[name](*[self.visit(e) for e in ctx.expr()])
        if name not in self.functions:
            raise Exception(f"Función '{name}' no definida")
        func = self.functions[name]
        args = [self.visit(e) for e in ctx.expr()]
        if len(args) != len(func["params"]): raise Exception("Número incorrecto de argumentos")
        key = (name, tuple(args))
        if key in self.memo: return self.memo[key]
        if len(self.scopes) > self.max_depth: raise Exception("Límite de recursión")
        self.scopes.append({})
        for p, a in zip(func["params"], args): self.current_scope()[p] = a
        try:
            self.visit(func["body"])
            result = None
        except ReturnException as r:
            result = r.value
        self.scopes.pop()
        self.memo[key] = result
        return result

    # ---------- Salida ----------
    def visitPrintStatement(self, ctx):
        def to_str(val):
            if isinstance(val, list): return "[" + ", ".join(to_str(v) for v in val) + "]"
            if isinstance(val, dict): return "{" + ", ".join(f"{to_str(k)}: {to_str(v)}" for k,v in val.items()) + "}"
            if isinstance(val, Stack): return repr(val)
            return str(val)
        print("TINTO >", " ".join(to_str(self.visit(e)) for e in ctx.expr()))

    # ---------- Literales simples ----------
    def visitNumero(self, ctx): return float(ctx.NUMBER().getText())
    def visitBooleano(self, ctx): return ctx.BOOLEAN().getText() == 'true'
    def visitCadena(self, ctx): return ctx.STRING().getText().strip('"')

    # ---------- Main ----------
    def main(self, input_file):
        lexer = TintoLexer(FileStream(input_file, encoding='utf-8'))
        parser = TintoParser(CommonTokenStream(lexer))
        try:
            tree = parser.program()
        except Exception as e:
            print(f"Error de sintaxis: {e}")
            return
        try:
            self.visit(tree)
        except Exception as e:
            print(f"Error durante la interpretación: {e}")
            import traceback; traceback.print_exc()

def main():
    if len(sys.argv) < 2:
        print("Uso: python interpreter.py <archivo.tinto>")
        sys.exit(1)
    TintoInterpreter().main(sys.argv[1])

if __name__ == '__main__':
    main()
