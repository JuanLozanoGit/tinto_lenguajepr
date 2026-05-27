import sys
import os
from antlr4 import *

sys.path.append(os.path.join(os.path.dirname(__file__), 'antlr_gen', 'grammar'))

from TintoLexer import TintoLexer
from TintoParser import TintoParser
from TintoVisitor import TintoVisitor

from tinto_ia import *

class ReturnException(Exception):

    def __init__(self, value):
        self.value = value


class Stack:

    def __init__(self):
        self._data = []

    def push(self, v):
        self._data.append(v)

    def pop(self):

        if not self._data:
            raise Exception("pop de pila vacía")

        return self._data.pop()

    def top(self):

        if not self._data:
            raise Exception("top de pila vacía")

        return self._data[-1]

    def is_empty(self):
        return len(self._data) == 0

    def size(self):
        return len(self._data)

    def __repr__(self):
        return f"stack({self._data})"


class TintoInterpreter(TintoVisitor):

    def __init__(self):

        super().__init__()

        self.globals = {

            "PI": 3.141592653589793,
            "E": 2.718281828459045,

            "stack": lambda: Stack(),
            "push": lambda s, v: s.push(v),
            "pop": lambda s: s.pop(),
            "top": lambda s: s.top(),
            "isEmpty": lambda s: s.is_empty(),
            "size": lambda s: s.size(),

            "crear": self.crear_archivo,
            "escribir": self.escribir_archivo,
            "agregar": self.agregar_archivo,
            "leer": self.leer_archivo,

            "linearRegression": tinto_linearRegression,
            "logisticRegression": tinto_logisticRegression,
            "perceptron": tinto_perceptron,
            "kNN": tinto_kNN,
            "kMeans": tinto_kMeans,

            "normalizar": tinto_normalizar,
            "errorCuadratico": tinto_errorCuadratico,
            "accuracy": tinto_accuracy,

            "crearRed": tinto_crearRed,
            "entrenar": tinto_entrenar,
            "predecir": tinto_predecir,

            "relu": tinto_relu,
            "sigmoid": tinto_sigmoid,
            "tanhAct": tinto_tanh,

            "sentimiento": tinto_sentimiento,
            "predecirSerie": tinto_predecirSerie,
            "similitud": tinto_similitud,
            "clasificarNumero": tinto_clasificarNumero
        }

        self.scopes = [self.globals]
        self.functions = {}
        self.memo = {}
        self.max_depth = 1000

    def current_scope(self):
        return self.scopes[-1]

    # =========================================================
    # ARCHIVOS TXT
    # =========================================================

    def crear_archivo(self, nombre):

        with open(nombre, "w", encoding="utf-8") as f:
            pass

        return None

    def escribir_archivo(self, nombre, texto):

        with open(nombre, "w", encoding="utf-8") as f:
            f.write(str(texto))

        return None

    def agregar_archivo(self, nombre, texto):

        with open(nombre, "a", encoding="utf-8") as f:
            f.write(str(texto))

        return None

    def leer_archivo(self, nombre):

        with open(nombre, "r", encoding="utf-8") as f:
            return f.read()

    # =========================================================
    # FUNCIONES MATEMÁTICAS
    # =========================================================

    def factorial(self, n):

        r = 1

        for i in range(1, int(n) + 1):
            r *= i

        return r

    def seno(self, grados):

        x = grados * self.globals["PI"] / 180
        res = 0

        for n in range(10):

            signo = -1 if n % 2 else 1
            res += signo * (x ** (2 * n + 1)) / self.factorial(2 * n + 1)

        return res

    def coseno(self, grados):

        x = grados * self.globals["PI"] / 180
        res = 0

        for n in range(10):

            signo = -1 if n % 2 else 1
            res += signo * (x ** (2 * n)) / self.factorial(2 * n)

        return res

    def tangente(self, grados):

        c = self.coseno(grados)

        if c == 0:
            raise Exception("Tangente indefinida")

        return self.seno(grados) / c

    def raiz(self, n):

        if n < 0:
            raise Exception("Raíz de negativo")

        if n == 0:
            return 0

        x = n

        for _ in range(20):
            x = (x + n / x) / 2

        return x

    def logaritmo(self, n):

        if n <= 0:
            raise Exception("Logaritmo inválido")

        x = 1

        for _ in range(20):

            ex = self.globals["E"] ** x
            x = x - (ex - n) / ex

        return x

    # =========================================================
    # PARSER AUXILIAR
    # =========================================================

    def _eval_expr(self, text):

        lexer = TintoLexer(InputStream(text))
        parser = TintoParser(CommonTokenStream(lexer))

        return self.visit(parser.expr())

    def _split_outer(self, text, sep, start, end):

        parts = []
        cur = ""
        nivel = 0
        dentro_string = False

        for ch in text:

            if ch == '"' and (len(cur) == 0 or cur[-1] != '\\'):
                dentro_string = not dentro_string

            if ch in '{[' and not dentro_string:
                nivel += 1

            if ch in '}]' and not dentro_string:
                nivel -= 1

            if ch == sep and nivel == 0 and not dentro_string:

                parts.append(cur.strip())
                cur = ""

            else:
                cur += ch

        if cur.strip():
            parts.append(cur.strip())

        return parts

    # =========================================================
    # LISTAS / DICCIONARIOS
    # =========================================================

    def visitListaLiteral(self, ctx):

        inner = ctx.getText()[1:-1].strip()

        if not inner:
            return []

        return [
            self._eval_expr(elem)
            for elem in self._split_outer(inner, ',', '[', ']')
        ]

    def visitDiccionarioLiteral(self, ctx):

        inner = ctx.getText()[1:-1].strip()

        if not inner:
            return {}

        d = {}

        for par in self._split_outer(inner, ',', '{', '}'):

            if ':' not in par:
                raise Exception(f"Par inválido: {par}")

            k, v = par.split(':', 1)

            d[self._eval_expr(k)] = self._eval_expr(v)

        return d

    def visitMatriz(self, ctx):

        inner = ctx.getText()[1:-1].strip()

        if not inner:
            return []

        matriz = []

        for fila in self._split_outer(inner, ';', '[', ']'):

            matriz.append([
                self._eval_expr(elem)
                for elem in self._split_outer(fila, ',', '[', ']')
            ])

        return matriz

    # =========================================================
    # BLOQUES
    # =========================================================

    def visitProgram(self, ctx):

        for stmt in ctx.statement():
            self.visit(stmt)

    def visitBloque(self, ctx):

        for stmt in ctx.statement():
            self.visit(stmt)

    # =========================================================
    # VARIABLES
    # =========================================================

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

    def visitVariable(self, ctx):

        name = ctx.ID().getText()

        for scope in reversed(self.scopes):

            if name in scope:
                return scope[name]

        raise NameError(f"Variable '{name}' no definida")

    # =========================================================
    # CONTROL
    # =========================================================

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

        for i in range(start, end + 1):

            self.current_scope()[var] = i
            self.visit(ctx.bloque())

    def visitReturnStatement(self, ctx):

        raise ReturnException(self.visit(ctx.expr()))
    
    def visitIndexacion(self, ctx):

        lista = self.visit(ctx.expr(0))
        indice = self.visit(ctx.expr(1))

        if not isinstance(lista, list):
            raise Exception("Intentando indexar algo que no es una lista")

        indice = int(indice)

        if indice < 0 or indice >= len(lista):
            raise Exception("Índice fuera de rango")

        return lista[indice]


    def visitAssignmentIndexed(self, ctx):

        nombre = ctx.ID().getText()

        indice = int(self.visit(ctx.expr(0)))
        valor = self.visit(ctx.expr(1))

        lista = None

    # Buscar la variable desde el scope interno hacia afuera
        for scope in reversed(self.scopes):

            if nombre in scope:
                lista = scope[nombre]
                break

        if lista is None:
            raise Exception(f"Variable '{nombre}' no definida")

        if not isinstance(lista, list):
            raise Exception(f"{nombre} no es una lista")

        if indice < 0 or indice >= len(lista):
            raise Exception("Índice fuera de rango")

        lista[indice] = valor

        return valor
    # =========================================================
    # OPERADORES
    # =========================================================

    def visitNegativo(self, ctx):

        return -self.visit(ctx.expr())

    def visitPotencia(self, ctx):

        return self.visit(ctx.expr(0)) ** self.visit(ctx.expr(1))

    def visitMulDivMod(self, ctx):

        l = self.visit(ctx.expr(0))
        r = self.visit(ctx.expr(1))

        op = ctx.op.text

        if op == '*':
            return l * r

        if op == '/':
            return l / r

        return l % r

    def visitSumaResta(self, ctx):

        l = self.visit(ctx.expr(0))
        r = self.visit(ctx.expr(1))

        if ctx.op.text == '+':
            return l + r

        return l - r

    def visitComparacion(self, ctx):

        l = self.visit(ctx.expr(0))
        r = self.visit(ctx.expr(1))

        op = ctx.op.text

        ops = {

            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '<': lambda a, b: a < b,
            '>': lambda a, b: a > b,
            '<=': lambda a, b: a <= b,
            '>=': lambda a, b: a >= b
        }

        return ops[op](l, r)

    def visitParentesis(self, ctx):

        return self.visit(ctx.expr())

    def visitNot(self, ctx):

        return not self.visit(ctx.expr())

    def visitAnd(self, ctx):

        return self.visit(ctx.expr(0)) and self.visit(ctx.expr(1))

    def visitOr(self, ctx):

        return self.visit(ctx.expr(0)) or self.visit(ctx.expr(1))

    # =========================================================
    # FUNCIONES
    # =========================================================

    def visitFunctionDeclaration(self, ctx):

        name = ctx.ID().getText()

        params = []

        if ctx.parameters():
            params = [p.getText() for p in ctx.parameters().ID()]

        self.functions[name] = {

            "params": params,
            "body": ctx.bloque()
        }

    def visitFunctionCall(self, ctx):

        name = ctx.ID().getText()

        args = [self.visit(e) for e in ctx.expr()]

    # FUNCIONES NATIVAS / GLOBALES
        if name in self.globals and callable(self.globals[name]):

            return self.globals[name](*args)

    # FUNCIONES DEL USUARIO
        if name not in self.functions:

            raise Exception(f"Función '{name}' no definida")

        func = self.functions[name]

        if len(args) != len(func["params"]):

            raise Exception("Número incorrecto de argumentos")

        self.scopes.append({})

        for p, a in zip(func["params"], args):

            self.current_scope()[p] = a

        try:

            self.visit(func["body"])
            result = None

        except ReturnException as r:

            result = r.value

        self.scopes.pop()

        return result

    # =========================================================
    # PRINT
    # =========================================================

    def visitPrintStatement(self, ctx):

        def to_str(val):

            if isinstance(val, list):

                return "[" + ", ".join(to_str(v) for v in val) + "]"

            if isinstance(val, dict):

                return "{" + ", ".join(
                    f"{to_str(k)}: {to_str(v)}"
                    for k, v in val.items()
                ) + "}"

            if isinstance(val, Stack):

                return repr(val)

            return str(val)

        print(
            "TINTO >",
            " ".join(to_str(self.visit(e)) for e in ctx.expr())
        )
        # ---------- Inteligencia Artificial ----------

    def visitRegLineal(self, ctx):
        X = self.visit(ctx.expr(0))
        Y = self.visit(ctx.expr(1))
        return self.globals["linearRegression"](X, Y)

    def visitRegLogistica(self, ctx):
        X = self.visit(ctx.expr(0))
        Y = self.visit(ctx.expr(1))
        return self.globals["logisticRegression"](X, Y)

    def visitPerceptron(self, ctx):
        X = self.visit(ctx.expr(0))
        Y = self.visit(ctx.expr(1))
        epocas = self.visit(ctx.expr(2))
        return self.globals["perceptron"](X, Y, epocas)

    def visitKNN(self, ctx):
        X = self.visit(ctx.expr(0))
        Y = self.visit(ctx.expr(1))
        k = self.visit(ctx.expr(2))
        return self.globals["kNN"](X, Y, k)

    def visitKMeans(self, ctx):
        datos = self.visit(ctx.expr(0))
        k = self.visit(ctx.expr(1))
        return self.globals["kMeans"](datos, k)

    def visitNormalizar(self, ctx):
        datos = self.visit(ctx.expr())
        return self.globals["normalizar"](datos)

    def visitErrorCuadratico(self, ctx):
        real = self.visit(ctx.expr(0))
        pred = self.visit(ctx.expr(1))
        return self.globals["errorCuadratico"](real, pred)

    def visitAccuracy(self, ctx):
        real = self.visit(ctx.expr(0))
        pred = self.visit(ctx.expr(1))
        return self.globals["accuracy"](real, pred)

    def visitCrearRed(self, ctx):
        capas = self.visit(ctx.expr())
        return self.globals["crearRed"](capas)

    def visitEntrenar(self, ctx):
        red = self.visit(ctx.expr(0))
        X = self.visit(ctx.expr(1))
        Y = self.visit(ctx.expr(2))
        epocas = self.visit(ctx.expr(3))
        return self.globals["entrenar"](red, X, Y, epocas)

    def visitPredecir(self, ctx):
        modelo = self.visit(ctx.expr(0))
        x = self.visit(ctx.expr(1))
        return self.globals["predecir"](modelo, x)

    def visitRelu(self, ctx):
        x = self.visit(ctx.expr())
        return self.globals["relu"](x)

    def visitSigmoid(self, ctx):
        x = self.visit(ctx.expr())
        return self.globals["sigmoid"](x)

    def visitTanhAct(self, ctx):
        x = self.visit(ctx.expr())
        return self.globals["tanhAct"](x)

    def visitSentimiento(self, ctx):
        texto = self.visit(ctx.expr())
        return self.globals["sentimiento"](texto)

    def visitPredecirSerie(self, ctx):
        lista = self.visit(ctx.expr())
        return self.globals["predecirSerie"](lista)

    def visitSimilitud(self, ctx):
        a = self.visit(ctx.expr(0))
        b = self.visit(ctx.expr(1))
        return self.globals["similitud"](a, b)

    def visitClasificarNumero(self, ctx):
        segmentos = self.visit(ctx.expr())
        return self.globals["clasificarNumero"](segmentos)
    # =========================================================
    # LITERALES
    # =========================================================

    def visitNumero(self, ctx):

        return float(ctx.NUMBER().getText())

    def visitBooleano(self, ctx):

        return ctx.BOOLEAN().getText() == 'true'

    def visitCadena(self, ctx):

        return ctx.STRING().getText().strip('"')

    # =========================================================
    # MAIN
    # =========================================================

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

            import traceback
            traceback.print_exc()


def main():

    if len(sys.argv) < 2:

        print("Uso: python interpreter.py <archivo.tinto>")
        sys.exit(1)

    TintoInterpreter().main(sys.argv[1])


if __name__ == '__main__':
    main()
