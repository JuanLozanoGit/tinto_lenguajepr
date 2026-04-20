import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from antlr4 import *
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier

# Importar ANTLR
sys.path.append(os.path.join(os.path.dirname(__file__), 'antlr_gen'))
from antlr_gen.TintoLexer import TintoLexer
from antlr_gen.TintoParser import TintoParser
from antlr_gen.TintoVisitor import TintoVisitor

class TintoInterpreter(TintoVisitor):
    def __init__(self):
        super().__init__()
        self.globals = {"PI": np.pi, "E": np.e}
        self.scopes = [self.globals]

    def current_scope(self):
        return self.scopes[-1]

    # --- Estructura ---
    def visitProgram(self, ctx):
        for stmt in ctx.statement():
            self.visit(stmt)

    def visitBloque(self, ctx):
        self.scopes.append({}) # Entrar a nuevo scope
        for stmt in ctx.statement():
            self.visit(stmt)
        self.scopes.pop() # Salir de scope

    # --- Variables ---
    def visitVariableDeclaration(self, ctx):
        name = ctx.ID().getText()
        value = self.visit(ctx.expr())
        self.current_scope()[name] = value
        return value

    def visitAssignment(self, ctx):
        name = ctx.ID().getText()
        value = self.visit(ctx.expr())
        # Buscar en qué scope existe la variable para actualizarla
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

    # --- Control de Flujo ---
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
        start = self.visit(ctx.expr(0))
        end = self.visit(ctx.expr(1))
        # Ciclo simple
        for i in range(int(start), int(end) + 1):
            self.current_scope()[var] = i
            self.visit(ctx.bloque())

    # --- Operaciones ---
    def visitPotencia(self, ctx):
        return np.power(self.visit(ctx.expr(0)), self.visit(ctx.expr(1)))

    def visitMulDivMod(self, ctx):
        l, r = self.visit(ctx.expr(0)), self.visit(ctx.expr(1))
        op = ctx.op.text
        if op == '*': return l @ r if isinstance(l, np.ndarray) else l * r
        if op == '/': return l / r
        return l % r

    def visitSumaResta(self, ctx):
        l, r = self.visit(ctx.expr(0)), self.visit(ctx.expr(1))
        return l + r if ctx.op.text == '+' else l - r

    def visitComparacion(self, ctx):
        l, r = self.visit(ctx.expr(0)), self.visit(ctx.expr(1))
        op = ctx.op.text
        ops = {'==': lambda a,b: a==b, '!=': lambda a,b: a!=b, '<': lambda a,b: a<b, '>': lambda a,b: a>b, '<=': lambda a,b: a<=b, '>=': lambda a,b: a>=b}
        return ops[op](l, r)

    # --- Funciones y IO ---
    def visitPrintStatement(self, ctx):
        args = [str(self.visit(e)) for e in ctx.expr()]
        print("TINTO >", " ".join(args))

    def visitPlotStatement(self, ctx):
        if len(ctx.expr()) == 2:
            plt.scatter(self.visit(ctx.expr(0)), self.visit(ctx.expr(1)))
        else:
            plt.plot(self.visit(ctx.expr(0)))
        plt.show()

    # --- Literales ---
    def visitNumero(self, ctx): return float(ctx.NUMBER().getText())
    def visitBooleano(self, ctx): return ctx.BOOLEAN().getText() == 'true'
    def visitCadena(self, ctx): return ctx.STRING().getText().strip('"')

def main():
    if len(sys.argv) < 2:
        print("Uso: python interpreter.py <archivo.tinto>")
        return
    
    lexer = TintoLexer(FileStream(sys.argv[1], encoding='utf-8'))
    stream = CommonTokenStream(lexer)
    parser = TintoParser(stream)
    tree = parser.program()
    
    interpreter = TintoInterpreter()
    interpreter.visit(tree)

if __name__ == '__main__':
    main()
        return int(ctx.getText())

    def visitID(self, ctx):
        return self.variables.get(ctx.getText(), 0)

    # --- Lógica de Bloques ---
    
    def visitBlock(self, ctx):
        for stmt in ctx.statement():
            self.visit(stmt)
        parser = TintoParser(stream)

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
    
    input_file = sys.argv[1]
    interpreter = TintoInterpreter()
    interpreter.main(input_file)


if __name__ == '__main__':
    main()
