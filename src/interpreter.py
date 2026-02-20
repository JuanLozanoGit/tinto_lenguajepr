import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Soporte para encontrar archivos de ANTLR
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
        self.variables = {"PI": np.pi, "E": np.e}

    def visitProgram(self, ctx: TintoParser.ProgramContext):
        for stmt in ctx.statement():
            self.visit(stmt)

    # --- Manejo de Bloques (Crucial para If/While) ---
    def visitBlock(self, ctx: TintoParser.BlockContext):
        last_val = None
        for stmt in ctx.statement():
            last_val = self.visit(stmt)
        return last_val

    def visitIfStatement(self, ctx: TintoParser.IfStatementContext):
        condition = self.visit(ctx.expr())
        if condition:
            return self.visit(ctx.block(0))
        elif ctx.block(1):
            return self.visit(ctx.block(1))

    def visitWhileStatement(self, ctx: TintoParser.WhileStatementContext):
        while self.visit(ctx.expr()):
            self.visit(ctx.block())

    # --- Expresiones ---
    def visitAssignment(self, ctx: TintoParser.AssignmentContext):
        name = ctx.ID().getText()
        value = self.visit(ctx.expr())
        self.variables[name] = value
        return value

    def visitSumaResta(self, ctx: TintoParser.SumaRestaContext):
        l, r = self.visit(ctx.expr(0)), self.visit(ctx.expr(1))
        return l + r if ctx.op.text == '+' else l - r

    def visitMultiplicacionDiv(self, ctx: TintoParser.MultiplicacionDivContext):
        l, r = self.visit(ctx.expr(0)), self.visit(ctx.expr(1))
        if ctx.op.text == '*':
            if isinstance(l, np.ndarray) and isinstance(r, np.ndarray):
                return np.matmul(l, r) # Multiplicación matricial correcta
            return l * r
        if ctx.op.text == '/': return l / r
        return l % r

    def visitMatrizLiteral(self, ctx: TintoParser.MatrizLiteralContext):
        return np.array([self.visit(row) for row in ctx.row()])

    # --- Funciones Especiales y Deep Learning ---
    def visitLlamadaFuncion(self, ctx: TintoParser.LlamadaFuncionContext):
        func = ctx.ID().getText()
        args = [self.visit(a) for a in ctx.argList().expr()] if ctx.argList() else []

        # Matemáticas y Matrices
        if func == "raiz": return np.sqrt(args[0])
        if func == "seno": return np.sin(args[0])
        if func == "coseno": return np.cos(args[0])
        if func == "transpuesta": return np.transpose(args[0])
        if func == "inversa": return np.linalg.inv(args[0])

        # Archivos
        if func == "leer": return pd.read_csv(args[0]).values
        if func == "escribir": 
            pd.DataFrame(args[1]).to_csv(args[0], index=False)
            return True

        # Deep Learning / ML
        if func == "regresion_lineal":
            model = LinearRegression().fit(args[0], args[1])
            return model.predict(args[0])
        
        if func == "regresion_logistica":
            model = LogisticRegression().fit(args[0], args[1].ravel())
            return model.predict(args[0])

        if func == "perceptron":
            mlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000)
            mlp.fit(args[0], args[1].ravel())
            return mlp.predict(args[0])

        return None

    def visitPrintStatement(self, ctx: TintoParser.PrintStatementContext):
        print(f"☕ TINTO > {self.visit(ctx.expr())}")

    def visitGraphStatement(self, ctx: TintoParser.GraphStatementContext):
        plt.plot(self.visit(ctx.expr(0)), self.visit(ctx.expr(1)), 'o-')
        plt.show()
