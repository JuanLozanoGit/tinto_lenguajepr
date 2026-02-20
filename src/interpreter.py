import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Forzar que encuentre los archivos generados si no están en el path
sys.path.append(os.path.join(os.path.dirname(__file__), 'antlr_gen'))

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier

# Importación única y segura
try:
    from TintoVisitor import TintoVisitor
    from TintoParser import TintoParser
except ImportError:
    from antlr_gen.TintoVisitor import TintoVisitor
    from antlr_gen.TintoParser import TintoParser

class TintoInterpreter(TintoVisitor):
    def __init__(self):
        super().__init__()
        # Diccionario de variables (memoria del lenguaje)
        self.variables = {
            "PI": np.pi,
            "E": np.e
        }

    # --- Puntos de Entrada ---
    def visitProgram(self, ctx: TintoParser.ProgramContext):
        results = []
        for stmt in ctx.statement():
            results.append(self.visit(stmt))
        return results

    # --- Estructuras de Control ---
    def visitIfStatement(self, ctx: TintoParser.IfStatementContext):
        condition = self.visit(ctx.expr())
        if condition:
            for stmt in ctx.statement()[:1]: # Ejecuta bloque IF
                self.visit(stmt)
        elif ctx.statement(1): # Si existe bloque SINO
            self.visit(ctx.statement(1))

    def visitWhileStatement(self, ctx: TintoParser.WhileStatementContext):
        while self.visit(ctx.expr()):
            for stmt in ctx.statement():
                self.visit(stmt)

    # --- Expresiones y Aritmética ---
    def visitAssignment(self, ctx: TintoParser.AssignmentContext):
        name = ctx.ID().getText()
        value = self.visit(ctx.expr())
        self.variables[name] = value
        return value

    def visitNumber(self, ctx: TintoParser.NumberContext):
        return float(ctx.NUMBER().getText())

    def visitString(self, ctx: TintoParser.StringContext):
        return ctx.STRING().getText().strip('"')

    def visitVariable(self, ctx: TintoParser.VariableContext):
        name = ctx.ID().getText()
        if name in self.variables:
            return self.variables[name]
        print(f"Error: Variable '{name}' no definida.")
        return 0

    def visitSumaResta(self, ctx: TintoParser.SumaRestaContext):
        l = self.visit(ctx.expr(0))
        r = self.visit(ctx.expr(1))
        return l + r if ctx.op.text == '+' else l - r

    def visitMultiplicacionDiv(self, ctx: TintoParser.MultiplicacionDivContext):
        l = self.visit(ctx.expr(0))
        r = self.visit(ctx.expr(1))
        if ctx.op.text == '*':
            # Soporte nativo para multiplicación de matrices (dot product)
            if isinstance(l, np.ndarray) or isinstance(r, np.ndarray):
                return np.dot(l, r)
            return l * r
        return l / r

    def visitPotencia(self, ctx: TintoParser.PotenciaContext):
        base = self.visit(ctx.expr(0))
        exp = self.visit(ctx.expr(1))
        return np.power(base, exp)

    # --- Matrices ---
    def visitMatrizLiteral(self, ctx: TintoParser.MatrizLiteralContext):
        return np.array([self.visit(row) for row in ctx.row()])

    def visitRow(self, ctx: TintoParser.RowContext):
        return [self.visit(e) for e in ctx.expr()]

    # --- Deep Learning y Funciones ---
    def visitLlamadaFuncion(self, ctx: TintoParser.LlamadaFuncionContext):
        func = ctx.ID().getText()
        args = [self.visit(a) for a in ctx.argList().expr()] if ctx.argList() else []

        # Operaciones Matemáticas
        if func == "seno": return np.sin(args[0])
        if func == "coseno": return np.cos(args[0])
        if func == "raiz": return np.sqrt(args[0])
        
        # Archivos
        if func == "leer": 
            return pd.read_csv(args[0]).values
        
        # Aprendizaje Profundo
        if func == "regresion":
            X, y = np.array(args[0]), np.array(args[1])
            model = LinearRegression().fit(X, y)
            return model.predict(X)
            
        if func == "perceptron":
            X, y = np.array(args[0]), np.array(args[1])
            # Perceptrón Multicapa (MLP)
            mlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000)
            mlp.fit(X, y.ravel())
            return mlp.predict(X)
            
        return None

    # --- Salida Visual ---
    def visitPrintStatement(self, ctx: TintoParser.PrintStatementContext):
        val = self.visit(ctx.expr())
        print(f"☕ TINTO > {val}")

    def visitGraphStatement(self, ctx: TintoParser.GraphStatementContext):
        x = self.visit(ctx.expr(0))
        y = self.visit(ctx.expr(1))
        plt.figure(figsize=(8,5))
        plt.plot(x, y, 'o-', color='brown', label='Datos TINTO')
        plt.title("Gráfica de Red Neuronal / Datos")
        plt.grid(True)
        plt.legend()
        plt.show()