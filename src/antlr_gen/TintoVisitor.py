# Generated from grammar/Tinto.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .TintoParser import TintoParser
else:
    from TintoParser import TintoParser


# FUNCIONES MATEMÁTICAS MANUALES (SEN, COS, TAN)


def factorial(n):
    """Calcula el factorial de n"""
    if n <= 1:
        return 1
    return n * factorial(n-1)

def seno_manual(x):
    """
    Calcula el seno de x usando serie de Taylor
    sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...
    """
    pi = 3.141592653589793
    
  
    x = x % (2 * pi)
    if x > pi:
        x -= 2 * pi
    
    termino = x
    resultado = x
    n = 1
    
    while abs(termino) > 1e-10:
        termino *= -x * x / ((2*n) * (2*n + 1))
        resultado += termino
        n += 1
    
    return resultado

def coseno_manual(x):
    """
    Calcula el coseno de x usando serie de Taylor
    cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! + ...
    """
    pi = 3.141592653589793
    
    
    x = x % (2 * pi)
    if x > pi:
        x -= 2 * pi
    
    termino = 1
    resultado = 1
    n = 1
    
    while abs(termino) > 1e-10:
        termino *= -x * x / ((2*n-1) * (2*n))
        resultado += termino
        n += 1
    
    return resultado

def tangente_manual(x):
    """
    Calcula la tangente de x, devuelve "indefinido" para asíntotas
    """
    pi = 3.141592653589793
    
    # Reducir x al rango [-π, π]
    x = x % (2 * pi)
    if x > pi:
        x -= 2 * pi
    
    # TOLERANCIA MÁS GRANDE (antes era 1e-10)
    tolerancia = 1e-6  # <--- CAMBIADO A 1e-6
    
    # Verificar si estamos en π/2 (90°) o -π/2 (-90°)
    if abs(abs(x) - pi/2) < tolerancia:
        return "indefinido"
    
    cos = coseno_manual(x)
    return seno_manual(x) / cos


class TintoVisitor(ParseTreeVisitor):
    
    def __init__(self):
        self.variables = {"PI": 3.141592653589793, "E": 2.718281828459045}
        self.scopes = [self.variables]  # Pila de ámbitos

    def current_scope(self):
        return self.scopes[-1]

    # Visit a parse tree produced by TintoParser#program.
    def visitProgram(self, ctx: TintoParser.ProgramContext):
        for stmt in ctx.statement():
            self.visit(stmt)

    # Visit a parse tree produced by TintoParser#statement.
    def visitStatement(self, ctx: TintoParser.StatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by TintoParser#variableDeclaration.
    def visitVariableDeclaration(self, ctx: TintoParser.VariableDeclarationContext):
        name = ctx.ID().getText()
        value = self.visit(ctx.expr())
        self.current_scope()[name] = value
        return value

    # Visit a parse tree produced by TintoParser#type.
    def visitType(self, ctx: TintoParser.TypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by TintoParser#assignment.
    def visitAssignment(self, ctx: TintoParser.AssignmentContext):
        name = ctx.ID().getText()
        value = self.visit(ctx.expr())
        self.current_scope()[name] = value
        return value

    # Visit a parse tree produced by TintoParser#ifStatement.
    def visitIfStatement(self, ctx: TintoParser.IfStatementContext):
        if self.visit(ctx.expr()):
            self.visit(ctx.bloque(0))
        elif len(ctx.bloque()) > 1:
            self.visit(ctx.bloque(1))

    # Visit a parse tree produced by TintoParser#whileStatement.
    def visitWhileStatement(self, ctx: TintoParser.WhileStatementContext):
        while self.visit(ctx.expr()):
            self.visit(ctx.bloque())

    # Visit a parse tree produced by TintoParser#forStatement.
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

    # Visit a parse tree produced by TintoParser#bloque.
    def visitBloque(self, ctx: TintoParser.BloqueContext):
        self.scopes.append({})
        for stmt in ctx.statement():
            self.visit(stmt)
        self.scopes.pop()

    # Visit a parse tree produced by TintoParser#returnStatement.
    def visitReturnStatement(self, ctx: TintoParser.ReturnStatementContext):
        return self.visit(ctx.expr())

    # Visit a parse tree produced by TintoParser#printStatement.
    def visitPrintStatement(self, ctx: TintoParser.PrintStatementContext):
        values = [self.visit(e) for e in ctx.expr()]
        print(" TINTO >", *values)

    # Visit a parse tree produced by TintoParser#plotStatement.
    def visitPlotStatement(self, ctx: TintoParser.PlotStatementContext):
        print(" Función plot deshabilitada en versión educativa")
        print("Datos a graficar:", [self.visit(e) for e in ctx.expr()])

    # Visit a parse tree produced by TintoParser#mulDivMod.
    def visitMulDivMod(self, ctx: TintoParser.MulDivModContext):
        l = self.visit(ctx.expr(0))
        r = self.visit(ctx.expr(1))
        op = ctx.op.text
        
        if op == '*':
            return l * r
        elif op == '/':
            return l / r
        else:  # '%'
            return l % r

    # Visit a parse tree produced by TintoParser#funcMat.  <--- MODIFICADO
    def visitFuncMat(self, ctx: TintoParser.FuncMatContext):
        func = ctx.getChild(0).getText()
        arg = self.visit(ctx.expr())
        
        if func == 'sin':
            return seno_manual(arg)
        elif func == 'cos':
            return coseno_manual(arg)
        elif func == 'tan':
            return tangente_manual(arg)
        else:
            raise NotImplementedError(f"Función '{func}' no implementada aún")

    # Visit a parse tree produced by TintoParser#regLogistica.
    def visitRegLogistica(self, ctx: TintoParser.RegLogisticaContext):
        raise NotImplementedError("Regresión logística no implementada")

    # Visit a parse tree produced by TintoParser#or.
    def visitOr(self, ctx: TintoParser.OrContext):
        return self.visit(ctx.expr(0)) or self.visit(ctx.expr(1))

    # Visit a parse tree produced by TintoParser#perceptron.
    def visitPerceptron(self, ctx: TintoParser.PerceptronContext):
        raise NotImplementedError("Perceptrón no implementado")

    # Visit a parse tree produced by TintoParser#numero.
    def visitNumero(self, ctx: TintoParser.NumeroContext):
        return float(ctx.NUMBER().getText())

    # Visit a parse tree produced by TintoParser#cadena.
    def visitCadena(self, ctx: TintoParser.CadenaContext):
        return ctx.STRING().getText().strip('"')

    # Visit a parse tree produced by TintoParser#regLineal.
    def visitRegLineal(self, ctx: TintoParser.RegLinealContext):
        raise NotImplementedError("Regresión lineal no implementada")

    # Visit a parse tree produced by TintoParser#matriz.
    def visitMatriz(self, ctx: TintoParser.MatrizContext):
        filas = []
        for fila_ctx in ctx.fila():
            fila = [self.visit(e) for e in fila_ctx.expr()]
            filas.append(fila)
        return filas

    # Visit a parse tree produced by TintoParser#parentesis.
    def visitParentesis(self, ctx: TintoParser.ParentesisContext):
        return self.visit(ctx.expr())

    # Visit a parse tree produced by TintoParser#sumaResta.
    def visitSumaResta(self, ctx: TintoParser.SumaRestaContext):
        l = self.visit(ctx.expr(0))
        r = self.visit(ctx.expr(1))
        return l + r if ctx.op.text == '+' else l - r

    # Visit a parse tree produced by TintoParser#potencia.
    def visitPotencia(self, ctx: TintoParser.PotenciaContext):
        base = self.visit(ctx.expr(0))
        exp = self.visit(ctx.expr(1))
        return base ** exp  # Usamos el operador nativo de Python

    # Visit a parse tree produced by TintoParser#not.
    def visitNot(self, ctx: TintoParser.NotContext):
        return not self.visit(ctx.expr())

    # Visit a parse tree produced by TintoParser#llamadaFuncion.
    def visitLlamadaFuncion(self, ctx: TintoParser.LlamadaFuncionContext):
        func_name = ctx.ID().getText()
        args = [self.visit(e) for e in ctx.expr()]
        
        # Funciones matemáticas en español
        if func_name == "seno":
            return seno_manual(args[0])
        elif func_name == "coseno":
            return coseno_manual(args[0])
        elif func_name == "tangente":
            return tangente_manual(args[0])
        else:
            raise ValueError(f"Función '{func_name}' no reconocida")

    # Visit a parse tree produced by TintoParser#booleano.
    def visitBooleano(self, ctx: TintoParser.BooleanoContext):
        return ctx.BOOLEAN().getText() == 'true'

    # Visit a parse tree produced by TintoParser#and.
    def visitAnd(self, ctx: TintoParser.AndContext):
        return self.visit(ctx.expr(0)) and self.visit(ctx.expr(1))

    # Visit a parse tree produced by TintoParser#transpuesta.
    def visitTranspuesta(self, ctx: TintoParser.TranspuestaContext):
        raise NotImplementedError("Transpuesta no implementada")

    # Visit a parse tree produced by TintoParser#variable.
    def visitVariable(self, ctx: TintoParser.VariableContext):
        name = ctx.ID().getText()
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        raise NameError(f"Variable '{name}' no definida")

    # Visit a parse tree produced by TintoParser#comparacion.
    def visitComparacion(self, ctx: TintoParser.ComparacionContext):
        l = self.visit(ctx.expr(0))
        r = self.visit(ctx.expr(1))
        op = ctx.op.text
        if op == '==': return l == r
        elif op == '!=': return l != r
        elif op == '<': return l < r
        elif op == '>': return l > r
        elif op == '<=': return l <= r
        else: return l >= r

    # Visit a parse tree produced by TintoParser#inversa.
    def visitInversa(self, ctx: TintoParser.InversaContext):
        raise NotImplementedError("Inversa no implementada")

    # Visit a parse tree produced by TintoParser#matrix.
    def visitMatrix(self, ctx: TintoParser.MatrixContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by TintoParser#fila.
    def visitFila(self, ctx: TintoParser.FilaContext):
        return [self.visit(e) for e in ctx.expr()]

    # Visit a parse tree produced by TintoParser#functionCall.
    def visitFunctionCall(self, ctx: TintoParser.FunctionCallContext):
        return self.visitChildren(ctx)


del TintoParser
