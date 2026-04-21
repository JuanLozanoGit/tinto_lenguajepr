    import sys
import os
from antlr4 import *

# Importar ANTLR
sys.path.append(os.path.join(os.path.dirname(__file__), 'antlr_gen'))
from antlr_gen.TintoLexer import TintoLexer
from antlr_gen.TintoParser import TintoParser
from antlr_gen.TintoVisitor import TintoVisitor


class ReturnException(Exception):
    def __init__(self, value):
        self.value = value


class TintoInterpreter(TintoVisitor):
    def __init__(self):
        super().__init__()
        self.globals = {"PI": 3.141592653589793, "E": 2.718281828459045}
        self.scopes = [self.globals]
        self.functions = {}

        #  NUEVO RECURSIVO
        self.memo = {}
        self.max_depth = 1000

    def current_scope(self):
        return self.scopes[-1]

    # --- Estructura ---
    def visitProgram(self, ctx):
        for stmt in ctx.statement():
            self.visit(stmt)

    def visitBloque(self, ctx):
        for stmt in ctx.statement():
            self.visit(stmt)

    # --- Variables ---
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
        for i in range(int(start), int(end) + 1):
            self.current_scope()[var] = i
            self.visit(ctx.bloque())

    def visitReturnStatement(self, ctx):
        value = self.visit(ctx.expr())
        raise ReturnException(value)

    # --- Operaciones ---
    def visitPotencia(self, ctx):
        return self.visit(ctx.expr(0)) ** self.visit(ctx.expr(1))

    def visitMulDivMod(self, ctx):
        l, r = self.visit(ctx.expr(0)), self.visit(ctx.expr(1))
        op = ctx.op.text
        if op == '*':
            return l * r
        if op == '/':
            return l / r
        return l % r

    def visitSumaResta(self, ctx):
        l, r = self.visit(ctx.expr(0)), self.visit(ctx.expr(1))
        return l + r if ctx.op.text == '+' else l - r

    def visitComparacion(self, ctx):
        l, r = self.visit(ctx.expr(0)), self.visit(ctx.expr(1))
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

    # --- Funciones ---
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

        if name not in self.functions:
            raise Exception(f"Función {name} no definida")

        func = self.functions[name]

        args = []
        if ctx.expr():
            args = [self.visit(e) for e in ctx.expr()]

        if len(args) != len(func["params"]):
            raise Exception("Número incorrecto de argumentos")

        #  Memoización
        key = (name, tuple(args))
        if key in self.memo:
            return self.memo[key]

        #  Límite de recursión
        if len(self.scopes) > self.max_depth:
            raise Exception("Límite de recursión alcanzado")

        # Crear scope
        self.scopes.append({})

        for param, arg in zip(func["params"], args):
            self.current_scope()[param] = arg

        try:
            self.visit(func["body"])
            result = None
        except ReturnException as r:
            result = r.value

        self.scopes.pop()

        # Guardar en memo
        self.memo[key] = result

        return result

    # --- IO ---
    def visitPrintStatement(self, ctx):
        args = [str(self.visit(e)) for e in ctx.expr()]
        print("TINTO >", " ".join(args))

    # --- Literales ---
    def visitNumero(self, ctx):
        return float(ctx.NUMBER().getText())

    def visitBooleano(self, ctx):
        return ctx.BOOLEAN().getText() == 'true'

    def visitCadena(self, ctx):
        return ctx.STRING().getText().strip('"')

    def main(self, input_file):
        lexer = TintoLexer(FileStream(input_file, encoding='utf-8'))
        stream = CommonTokenStream(lexer)
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
