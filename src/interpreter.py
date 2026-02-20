import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Asegurar carga de módulos generados en la carpeta antlr_gen
sys.path.append(os.path.join(os.path.dirname(__file__), 'antlr_gen'))

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier

try:
    # Intentar importar desde la carpeta generada
    from antlr_gen.TintoVisitor import TintoVisitor
    from antlr_gen.TintoParser import TintoParser
except ImportError:
    # Fallback por si la estructura de carpetas varía en el entorno de ejecución
    from TintoVisitor import TintoVisitor
    from TintoParser import TintoParser

class TintoInterpreter(TintoVisitor):
    def __init__(self):
        super().__init__()
        # Memoria global y tabla de constantes
        self.globals = {"PI": np.pi, "E": np.e}
        self.functions = {}
        self.scopes = [self.globals]

    def current_scope(self):
        return self.scopes[-1]

    # --- Ejecución de Bloques y Programa ---
    def visitProgram(self, ctx: TintoParser.ProgramContext):
        for stmt in ctx.statement():
            self.visit(stmt)

    def visitBlock(self, ctx: TintoParser.BlockContext):
        for stmt in ctx.statement():
            self.visit(stmt)

    # --- Estructuras de Control (Requisito: If, While, For) ---
    def visitIfStatement(self, ctx: TintoParser.IfStatementContext):
        condition = self.visit(ctx.expr())
        if condition:
            self.visit(ctx.block(0))
        elif ctx.block(1):
            self.visit(ctx.block(1))

    def visitWhileStatement(self, ctx: TintoParser.WhileStatementContext):
        while self.visit(ctx.expr()):
            self.visit(ctx.block())

    def visitForStatement(self, ctx: TintoParser.ForStatementContext):
        # Estructura esperada: for (asignacion; condicion; actualizacion)
        self.visit(ctx.assignment(0)) # Inicialización: i = 0
        while self.visit(ctx.expr(0)): # Condición: i < 10
            self.visit(ctx.block())
            self.visit(ctx.assignment(1)) # Actualización: i = i + 1

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

    # --- Operaciones Matemáticas y Matrices (Requisito: Aritmética Completa) ---
    def visitSumaResta(self, ctx: TintoParser.SumaRestaContext):
        l, r = self.visit(ctx.expr(0)), self.visit(ctx.expr(1))
        return l + r if ctx.op.text == '+' else l - r

    def visitMultiplicacionDiv(self, ctx: TintoParser.MultiplicacionDivContext):
        l, r = self.visit(ctx.expr(0)), self.visit(ctx.expr(1))
        if ctx.op.text == '*':
            # Multiplicación matricial automática si ambos son arrays
            if isinstance(l, np.ndarray) and isinstance(r, np.ndarray):
                return l @ r 
            return l * r
        if ctx.op.text == '/':
            return l / r
        return l % r

    def visitPotencia(self, ctx: TintoParser.PotenciaContext):
        # Requisito: Cálculo de raíces de la forma x^y (ej: x^0.5)
        return np.power(self.visit(ctx.expr(0)), self.visit(ctx.expr(1)))

    def visitMatrizLiteral(self, ctx: TintoParser.MatrizLiteralContext):
        return np.array([self.visit(row) for row in ctx.row()])

    # --- Funciones Nativas y Deep Learning ---
    def visitLlamadaFuncion(self, ctx: TintoParser.LlamadaFuncionContext):
        func_name = ctx.ID().getText()
        args = [self.visit(a) for a in ctx.argList().expr()] if ctx.argList() else []

        # 1. Requisito: Operaciones Trigonométricas y Matemáticas
        math_map = {
            "seno": np.sin, "coseno": np.cos, "tan": np.tan,
            "asen": np.arcsin, "acos": np.arccos, "atan": np.arctan,
            "raiz": np.sqrt, "log": np.log, "exp": np.exp,
            "transpuesta": np.transpose, "inversa": np.linalg.inv
        }
        if func_name in math_map:
            return math_map[func_name](args[0])

        # 2. Requisito: Manejo de Archivos
        if func_name == "leer": 
            return pd.read_csv(args[0]).values
        if func_name == "escribir":
            pd.DataFrame(args[1]).to_csv(args[0], index=False)
            return True

        # 3. Requisito: Modelos de Aprendizaje (Regresión, Perceptrón, ANN)
        if func_name == "regresion_lineal":
            model = LinearRegression().fit(args[0], args[1])
            return model.predict(args[0])
        
        if func_name == "regresion_logistica":
            model = LogisticRegression().fit(args[0], args[1].ravel())
            return model.predict(args[0])

        if func_name in ["perceptron", "red_neuronal", "clasificador"]:
            # MLP para agrupamiento/clasificación/predicción
            mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, activation='relu')
            mlp.fit(args[0], args[1].ravel())
            return mlp.predict(args[0])

        raise ValueError(f"Función '{func_name}' no reconocida.")

    # --- Requisito: Gráficas de datos ---
    def visitGraphStatement(self, ctx: TintoParser.GraphStatementContext):
        # Puede recibir un argumento (y) o dos (x, y)
        if len(ctx.expr()) == 2:
            x, y = self.visit(ctx.expr(0)), self.visit(ctx.expr(1))
            plt.scatter(x, y, color='brown', label='Datos')
        else:
            y = self.visit(ctx.expr(0))
            plt.plot(y, color='brown', label='Tendencia')
        
        plt.title("Visualización de Datos - Tinto DSL")
        plt.grid(True)
        plt.legend()
        plt.show()

    # --- Salida por consola ---
    def visitPrintStatement(self, ctx: TintoParser.PrintStatementContext):
        val = self.visit(ctx.expr())
        print(f"☕ TINTO > {val}")

    # --- Literales ---
    def visitNumber(self, ctx: TintoParser.NumberContext): 
        return float(ctx.NUMBER().getText())
    def visitString(self, ctx: TintoParser.StringContext): 
        return ctx.STRING().getText().strip('"')
    def visitRow(self, ctx: TintoParser.RowContext): 
        return [self.visit(e) for e in ctx.expr()]
