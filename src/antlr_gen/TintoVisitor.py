# Generated from grammar/Tinto.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .TintoParser import TintoParser
else:
    from TintoParser import TintoParser

# This class defines a complete generic visitor for a parse tree produced by TintoParser.

class TintoVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by TintoParser#program.
    def visitProgram(self, ctx:TintoParser.ProgramContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#statement.
    def visitStatement(self, ctx:TintoParser.StatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#variableDeclaration.
    def visitVariableDeclaration(self, ctx:TintoParser.VariableDeclarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#type.
    def visitType(self, ctx:TintoParser.TypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#assignment.
    def visitAssignment(self, ctx:TintoParser.AssignmentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#ifStatement.
    def visitIfStatement(self, ctx:TintoParser.IfStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#whileStatement.
    def visitWhileStatement(self, ctx:TintoParser.WhileStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#forStatement.
    def visitForStatement(self, ctx:TintoParser.ForStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#bloque.
    def visitBloque(self, ctx:TintoParser.BloqueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#returnStatement.
    def visitReturnStatement(self, ctx:TintoParser.ReturnStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#printStatement.
    def visitPrintStatement(self, ctx:TintoParser.PrintStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#plotStatement.
    def visitPlotStatement(self, ctx:TintoParser.PlotStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#mulDivMod.
    def visitMulDivMod(self, ctx:TintoParser.MulDivModContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#funcMat.
    def visitFuncMat(self, ctx:TintoParser.FuncMatContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#regLogistica.
    def visitRegLogistica(self, ctx:TintoParser.RegLogisticaContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#or.
    def visitOr(self, ctx:TintoParser.OrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#perceptron.
    def visitPerceptron(self, ctx:TintoParser.PerceptronContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#numero.
    def visitNumero(self, ctx:TintoParser.NumeroContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#cadena.
    def visitCadena(self, ctx:TintoParser.CadenaContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#regLineal.
    def visitRegLineal(self, ctx:TintoParser.RegLinealContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#matriz.
    def visitMatriz(self, ctx:TintoParser.MatrizContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#parentesis.
    def visitParentesis(self, ctx:TintoParser.ParentesisContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#sumaResta.
    def visitSumaResta(self, ctx:TintoParser.SumaRestaContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#potencia.
    def visitPotencia(self, ctx:TintoParser.PotenciaContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#not.
    def visitNot(self, ctx:TintoParser.NotContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#llamadaFuncion.
    def visitLlamadaFuncion(self, ctx:TintoParser.LlamadaFuncionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#booleano.
    def visitBooleano(self, ctx:TintoParser.BooleanoContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#and.
    def visitAnd(self, ctx:TintoParser.AndContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#transpuesta.
    def visitTranspuesta(self, ctx:TintoParser.TranspuestaContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#variable.
    def visitVariable(self, ctx:TintoParser.VariableContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#comparacion.
    def visitComparacion(self, ctx:TintoParser.ComparacionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#inversa.
    def visitInversa(self, ctx:TintoParser.InversaContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#matrix.
    def visitMatrix(self, ctx:TintoParser.MatrixContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#fila.
    def visitFila(self, ctx:TintoParser.FilaContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by TintoParser#functionCall.
    def visitFunctionCall(self, ctx:TintoParser.FunctionCallContext):
        return self.visitChildren(ctx)



del TintoParser