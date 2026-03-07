import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Asegurar que los módulos generados por ANTLR se encuentren
sys.path.append(os.path.join(os.path.dirname(__file__), 'antlr_gen'))

from antlr4 import *
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier

# Importaciones de los módulos generados (con try/except para flexibilidad)
try:
    from antlr_gen.TintoVisitor import TintoVisitor
    from antlr_gen.TintoParser import TintoParser
    from antlr_gen.TintoLexer import TintoLexer
except ImportError:
    from TintoVisitor import TintoVisitor
    from TintoParser import TintoParser
    from TintoLexer import TintoLexer

class TintoInterpreter(TintoVisitor):
    def __init__(self):
        super().__init__()
        self.globals = {"PI": np.pi, "E": np.e}
        self.functions = {}
        self.scopes = [self.globals]

    def current_scope(self):
        return self.scopes[-1]

    # --- Programa ---
    def visitProgram(self, ctx: TintoParser.ProgramContext):
        for stmt in ctx.statement():
            self.visit(stmt)

    # --- Bloque (ámbito local) ---
    def visitBloque(self, ctx: TintoParser.BloqueContext):
        self.scopes.append({})
        for stmt in ctx.statement():
            self.visit(stmt)
        self.scopes.pop()

    # --- Declaración de variables ---
    def visitVariableDeclaration(self, ctx: TintoParser.VariableDeclarationContext):
        name = ctx.ID().getText()
        value = self.visit(ctx.expr())
        self.current_scope()[name] = value
        return value

    # --- Asignación ---
    def visitAssignment(self, ctx: TintoParser.AssignmentContext):
        name = ctx.ID().getText()
        value = self.visit(ctx.expr())
        self.current_scope()[name] = value
        return value

    # --- Variable (identificador) ---
    def visitVariable(self, ctx: TintoParser.VariableContext):
        name = ctx.ID().getText()
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        raise NameError(f"Variable '{name}' no definida")

    # --- Estructuras de control ---
    def visitIfStatement(self, ctx: TintoParser.IfStatementContext):
        if self.visit(ctx.expr()):
            self.visit(ctx.bloque(0))
        elif ctx.bloque(1):
            self.visit(ctx.bloque(1))

    def visitWhileStatement(self, ctx: TintoParser.WhileStatementContext):
        while self.visit(ctx.expr()):
            self.visit(ctx.bloque())

    def visitForStatement(self, ctx: TintoParser.ForStatementContext):
        var = ctx.ID().getText()
        start = self.visit(ctx.expr(0))
        end = self.visit(ctx.expr(1))
        old = self.current_scope().get(var, None)
        step = 1 if start <= end else -1
        i = start
        while (step > 0 and i <= end) or (step < 0 and i >= end):
            self.current_scope()[var] = i
            self.visit(ctx.bloque())
            i += step
        if old is not None:
            self.current_scope()[var] = old
        elif var in self.current_scope():
            del self.current_scope()[var]

    # --- Retorno de función ---
    def visitReturnStatement(self, ctx: TintoParser.ReturnStatementContext):
        return self.visit(ctx.expr())

    # --- Salida por consola ---
    def visitPrintStatement(self, ctx: TintoParser.PrintStatementContext):
        values = [self.visit(e) for e in ctx.expr()]
        print("☕ TINTO >", *values)

    # --- Gráficas ---
    def visitPlotStatement(self, ctx: TintoParser.PlotStatementContext):
        if len(ctx.expr()) == 2:
            x = self.visit(ctx.expr(0))
            y = self.visit(ctx.expr(1))
            plt.scatter(x, y, color='brown', label='Datos')
        else:
            y = self.visit(ctx.expr(0))
            plt.plot(y, color='brown', label='Tendencia')
        plt.title("Visualización - Tinto DSL")
        plt.grid(True)
        plt.legend()
        plt.show()

    # --- Expresiones aritméticas y lógicas ---
    def visitPotencia(self, ctx: TintoParser.PotenciaContext):
        return np.power(self.visit(ctx.expr(0)), self.visit(ctx.expr(1)))

    def visitMulDivMod(self, ctx: TintoParser.MulDivModContext):
        l, r = self.visit(ctx.expr(0)), self.visit(ctx.expr(1))
        op = ctx.op.text
        if op == '*':
            if isinstance(l, np.ndarray) and isinstance(r, np.ndarray):
                return l @ r
            return l * r
        elif op == '/':
            return l / r
        else:  # '%'
            return l % r

    def visitSumaResta(self, ctx: TintoParser.SumaRestaContext):
        l, r = self.visit(ctx.expr(0)), self.visit(ctx.expr(1))
        return l + r if ctx.op.text == '+' else l - r

    def visitComparacion(self, ctx: TintoParser.ComparacionContext):
        l, r = self.visit(ctx.expr(0)), self.visit(ctx.expr(1))
        op = ctx.op.text
        if op == '==': return l == r
        elif op == '!=': return l != r
        elif op == '<': return l < r
        elif op == '>': return l > r
        elif op == '<=': return l <= r
        else: return l >= r

    def visitNot(self, ctx: TintoParser.NotContext):
        return not self.visit(ctx.expr())

    def visitAnd(self, ctx: TintoParser.AndContext):
        return self.visit(ctx.expr(0)) and self.visit(ctx.expr(1))

    def visitOr(self, ctx: TintoParser.OrContext):
        return self.visit(ctx.expr(0)) or self.visit(ctx.expr(1))

    def visitParentesis(self, ctx: TintoParser.ParentesisContext):
        return self.visit(ctx.expr())

    # --- Funciones matemáticas nativas ---
    def visitFuncMat(self, ctx: TintoParser.FuncMatContext):
        func = ctx.getChild(0).getText()
        arg = self.visit(ctx.expr())
        if func == 'sin': return np.sin(arg)
        elif func == 'cos': return np.cos(arg)
        elif func == 'tan': return np.tan(arg)
        elif func == 'sqrt': return np.sqrt(arg)
        elif func == 'log': return np.log(arg)
        else: raise ValueError(f"Función '{func}' no implementada")

    # --- Operaciones con matrices ---
    def visitTranspuesta(self, ctx: TintoParser.TranspuestaContext):
        return np.transpose(self.visit(ctx.expr()))

    def visitInversa(self, ctx: TintoParser.InversaContext):
        return np.linalg.inv(self.visit(ctx.expr()))

    # --- Regresiones y perceptrón ---
    def visitRegLineal(self, ctx: TintoParser.RegLinealContext):
        X = self.visit(ctx.expr(0))
        y = self.visit(ctx.expr(1))
        model = LinearRegression().fit(X, y)
        return model

    def visitRegLogistica(self, ctx: TintoParser.RegLogisticaContext):
        X = self.visit(ctx.expr(0))
        y = self.visit(ctx.expr(1))
        model = LogisticRegression().fit(X, y.ravel())
        return model

    def visitPerceptron(self, ctx: TintoParser.PerceptronContext):
        X = self.visit(ctx.expr(0))
        y = self.visit(ctx.expr(1))
        capas = self.visit(ctx.expr(2))
        if isinstance(capas, list):
            mlp = MLPClassifier(hidden_layer_sizes=tuple(capas), max_iter=1000)
        else:
            mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000)
        mlp.fit(X, y.ravel())
        return mlp

    # --- Matrices literales ---
    def visitMatriz(self, ctx: TintoParser.MatrizContext):
        filas = []
        # Obtener todos los contextos de tipo FilaContext dentro de la matriz
        for fila_ctx in ctx.getRuleContexts(TintoParser.FilaContext):
            fila = [self.visit(e) for e in fila_ctx.expr()]
            filas.append(fila)
        return np.array(filas)

    # --- Llamada a función definida por el usuario (o nativas en español) ---
    def visitLlamadaFuncion(self, ctx: TintoParser.LlamadaFuncionContext):
        func_name = ctx.ID().getText()
        args = [self.visit(e) for e in ctx.expr()]

        # Mapa de funciones matemáticas en español
        math_map = {
            "seno": np.sin, "coseno": np.cos, "tan": np.tan,
            "asen": np.arcsin, "acos": np.arccos, "atan": np.arctan,
            "raiz": np.sqrt, "log": np.log, "exp": np.exp,
            "transpuesta": np.transpose, "inversa": np.linalg.inv
        }
        if func_name in math_map:
            return math_map[func_name](args[0])

        # Archivos
        if func_name == "leer":
            return pd.read_csv(args[0]).values
        if func_name == "escribir":
            pd.DataFrame(args[1]).to_csv(args[0], index=False)
            return True

        # Modelos (regresiones y perceptrón como llamadas, para compatibilidad)
        if func_name == "regresion_lineal":
            model = LinearRegression().fit(args[0], args[1])
            return model.predict(args[0])
        if func_name == "regresion_logistica":
            model = LogisticRegression().fit(args[0], args[1].ravel())
            return model.predict(args[0])
        if func_name in ["perceptron", "red_neuronal", "clasificador"]:
            mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000)
            mlp.fit(args[0], args[1].ravel())
            return mlp.predict(args[0])

        # Función predecir (para usar con modelos entrenados)
        if func_name == "predecir":
            modelo = args[0]
            X = args[1]
            return modelo.predict(X)

        raise ValueError(f"Función '{func_name}' no reconocida")

    # --- Literales ---
    def visitNumero(self, ctx: TintoParser.NumeroContext):
        return float(ctx.NUMBER().getText())

    def visitBooleano(self, ctx: TintoParser.BooleanoContext):
        return ctx.BOOLEAN().getText() == 'true'

    def visitCadena(self, ctx: TintoParser.CadenaContext):
        return ctx.STRING().getText().strip('"')

    def visitFila(self, ctx: TintoParser.FilaContext):
        return [self.visit(e) for e in ctx.expr()]


def main():
    if len(sys.argv) < 2:
        print("Uso: python interpreter.py <archivo.tinto>")
        sys.exit(1)
    input_file = sys.argv[1]
    try:
        input_stream = FileStream(input_file, encoding='utf-8')
    except Exception as e:
        print(f"Error al abrir archivo: {e}")
        sys.exit(1)

    lexer = TintoLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = TintoParser(stream)

    # Manejo de errores de sintaxis
    try:
        tree = parser.program()
    except Exception as e:
        print(f"Error de sintaxis: {e}")
        sys.exit(1)

    interpreter = TintoInterpreter()
    try:
        interpreter.visit(tree)
    except Exception as e:
        print(f"Error durante la interpretación: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
