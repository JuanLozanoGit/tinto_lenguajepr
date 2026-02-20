import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Asegurar carga de módulos generados
sys.path.append(os.path.join(os.path.dirname(__file__), 'antlr_gen'))

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier

try:
    from TintoVisitor import TintoVisitor
    from TintoParser import TintoParser
except ImportError:
    from antlr_gen.TintoVisitor import TintoVisitor
    from antlr_gen.TintoParser import TintoParser

class TintoInterpreter(TintoVisitor):
    def __init__(self):
        super().__init__()
        # Memoria global y tabla de funciones
        self.globals = {"PI": np.pi, "E": np.e}
        self.functions = {}
        self.scopes = [self.globals]

    def current_scope(self):
        return self.scopes[-1]

    # --- Ejecución de Bloques ---
    def visitBlock(self, ctx: TintoParser.BlockContext):
        for stmt in ctx.statement():
            self.visit(stmt)

    def visitProgram(self, ctx: TintoParser.ProgramContext):
        for stmt in ctx.statement():
            self.visit(stmt)

    # --- Estructuras de Control ---
    def visitIfStatement(self, ctx: TintoParser.IfStatementContext):
        condition = self.visit(ctx.expr())
        if condition:
            self.visit(ctx.block(0))
        elif ctx.block(1):
            self.visit(ctx.block(1))

    def visitWhileStatement(self, ctx: TintoParser.WhileStatementContext):
        while self.visit(ctx.expr()):
            self.visit(ctx.block())

    # --- Variables y Asignación ---
    def visitAssignment(self, ctx: TintoParser.AssignmentContext):
        name = ctx.ID().getText()
        value = self.visit(ctx.expr())
        self.current_scope()[name] = value
        return value

    def visitVariable(self, ctx: TintoParser.VariableContext):
        name = ctx.ID().getText()
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        raise NameError(f"Error: Variable '{name}' no definida.")

    # --- Operaciones Matemáticas y Matrices ---
    def visitSumaResta(self, ctx: TintoParser.SumaRestaContext):
        l, r = self.visit(ctx.expr(0)), self.visit(ctx.expr(1))
        return l + r if ctx.op.text == '+' else l - r

    def visitMultiplicacionDiv(self, ctx: TintoParser.MultiplicacionDivContext):
        l, r = self.visit(ctx.expr(0)), self.visit(ctx.expr(1))
        if ctx.op.text == '*':
            if isinstance(l, np.ndarray) and isinstance(r, np.ndarray):
                return l @ r # Multiplicación matricial (dot product)
            return l * r
        return l / r if ctx.op.text == '/' else l % r

    def visitPotencia(self, ctx: TintoParser.PotenciaContext):
        return np.power(self.visit(ctx.expr(0)), self.visit(ctx.expr(1)))

    def visitMatrizLiteral(self, ctx: TintoParser.MatrizLiteralContext):
        return np.array([self.visit(row) for row in ctx.row()])

    # --- Funciones Nativas y de Deep Learning ---
    def visitLlamadaFuncion(self, ctx: TintoParser.LlamadaFuncionContext):
        func_name = ctx.ID().getText()
        args = [self.visit(a) for a in ctx.argList().expr()] if ctx.argList() else []

        # 1. Operaciones Matemáticas Requeridas
        math_map = {
            "seno": np.sin, "coseno": np.cos, "tan": np.tan,
            "raiz": np.sqrt, "log": np.log, "exp": np.exp,
            "transpuesta": np.transpose, "inversa": np.linalg.inv
        }
        if func_name in math_map:
            return math_map[func_name](args[0])

        # 2. Manejo de Archivos (Lectura/Escritura)
        if func_name == "leer": return pd.read_csv(args[0]).values
        if func_name == "escribir":
            pd.DataFrame(args[1]).to_csv(args[0], index=False)
            return True

        # 3. Modelos de Aprendizaje Automático
        if func_name == "regresion_lineal":
            model = LinearRegression().fit(args[0], args[1])
            return model.predict(args[0])
        
        if func_name == "regresion_logistica":
            model = LogisticRegression().fit(args[0], args[1].ravel())
            return model.predict(args[0])

        if func_name == "perceptron" or func_name == "red_neuronal":
            # Perceptrón Multicapa para Clasificación y Predicción
            mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, activation='relu')
            mlp.fit(args[0], args[1].ravel())
            return mlp.predict(args[0])

        raise ValueError(f"Función '{func_name}' no reconocida.")

    # --- Salida Visual ---
    def visitPrintStatement(self, ctx: TintoParser.PrintStatementContext):
        val = self.visit(ctx.expr())
        print(f"☕ TINTO > {val}")

    def visitGraphStatement(self, ctx: TintoParser.GraphStatementContext):
        x, y = self.visit(ctx.expr(0)), self.visit(ctx.expr(1))
        plt.figure(figsize=(10,6))
        plt.scatter(x, y, color='brown', label='Dataset')
        plt.title("Visualización de Datos - Proyecto Tinto")
        plt.grid(True)
        plt.legend()
        plt.show()

    # --- Literales ---
    def visitNumber(self, ctx: TintoParser.NumberContext): return float(ctx.NUMBER().getText())
    def visitString(self, ctx: TintoParser.StringContext): return ctx.STRING().getText().strip('"')
    def visitRow(self, ctx: TintoParser.RowContext): return [self.visit(e) for e in ctx.expr()]
