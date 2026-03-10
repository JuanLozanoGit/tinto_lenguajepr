# Generated from grammar/Tinto.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .TintoParser import TintoParser
else:
    from TintoParser import TintoParser

# This class defines a complete listener for a parse tree produced by TintoParser.
class TintoListener(ParseTreeListener):

    # Enter a parse tree produced by TintoParser#program.
    def enterProgram(self, ctx:TintoParser.ProgramContext):
        pass

    # Exit a parse tree produced by TintoParser#program.
    def exitProgram(self, ctx:TintoParser.ProgramContext):
        pass


    # Enter a parse tree produced by TintoParser#statement.
    def enterStatement(self, ctx:TintoParser.StatementContext):
        pass

    # Exit a parse tree produced by TintoParser#statement.
    def exitStatement(self, ctx:TintoParser.StatementContext):
        pass


    # Enter a parse tree produced by TintoParser#variableDeclaration.
    def enterVariableDeclaration(self, ctx:TintoParser.VariableDeclarationContext):
        pass

    # Exit a parse tree produced by TintoParser#variableDeclaration.
    def exitVariableDeclaration(self, ctx:TintoParser.VariableDeclarationContext):
        pass


    # Enter a parse tree produced by TintoParser#type.
    def enterType(self, ctx:TintoParser.TypeContext):
        pass

    # Exit a parse tree produced by TintoParser#type.
    def exitType(self, ctx:TintoParser.TypeContext):
        pass


    # Enter a parse tree produced by TintoParser#assignment.
    def enterAssignment(self, ctx:TintoParser.AssignmentContext):
        pass

    # Exit a parse tree produced by TintoParser#assignment.
    def exitAssignment(self, ctx:TintoParser.AssignmentContext):
        pass


    # Enter a parse tree produced by TintoParser#ifStatement.
    def enterIfStatement(self, ctx:TintoParser.IfStatementContext):
        pass

    # Exit a parse tree produced by TintoParser#ifStatement.
    def exitIfStatement(self, ctx:TintoParser.IfStatementContext):
        pass


    # Enter a parse tree produced by TintoParser#whileStatement.
    def enterWhileStatement(self, ctx:TintoParser.WhileStatementContext):
        pass

    # Exit a parse tree produced by TintoParser#whileStatement.
    def exitWhileStatement(self, ctx:TintoParser.WhileStatementContext):
        pass


    # Enter a parse tree produced by TintoParser#forStatement.
    def enterForStatement(self, ctx:TintoParser.ForStatementContext):
        pass

    # Exit a parse tree produced by TintoParser#forStatement.
    def exitForStatement(self, ctx:TintoParser.ForStatementContext):
        pass


    # Enter a parse tree produced by TintoParser#bloque.
    def enterBloque(self, ctx:TintoParser.BloqueContext):
        pass

    # Exit a parse tree produced by TintoParser#bloque.
    def exitBloque(self, ctx:TintoParser.BloqueContext):
        pass


    # Enter a parse tree produced by TintoParser#returnStatement.
    def enterReturnStatement(self, ctx:TintoParser.ReturnStatementContext):
        pass

    # Exit a parse tree produced by TintoParser#returnStatement.
    def exitReturnStatement(self, ctx:TintoParser.ReturnStatementContext):
        pass


    # Enter a parse tree produced by TintoParser#printStatement.
    def enterPrintStatement(self, ctx:TintoParser.PrintStatementContext):
        pass

    # Exit a parse tree produced by TintoParser#printStatement.
    def exitPrintStatement(self, ctx:TintoParser.PrintStatementContext):
        pass


    # Enter a parse tree produced by TintoParser#plotStatement.
    def enterPlotStatement(self, ctx:TintoParser.PlotStatementContext):
        pass

    # Exit a parse tree produced by TintoParser#plotStatement.
    def exitPlotStatement(self, ctx:TintoParser.PlotStatementContext):
        pass


    # Enter a parse tree produced by TintoParser#mulDivMod.
    def enterMulDivMod(self, ctx:TintoParser.MulDivModContext):
        pass

    # Exit a parse tree produced by TintoParser#mulDivMod.
    def exitMulDivMod(self, ctx:TintoParser.MulDivModContext):
        pass


    # Enter a parse tree produced by TintoParser#funcMat.
    def enterFuncMat(self, ctx:TintoParser.FuncMatContext):
        pass

    # Exit a parse tree produced by TintoParser#funcMat.
    def exitFuncMat(self, ctx:TintoParser.FuncMatContext):
        pass


    # Enter a parse tree produced by TintoParser#regLogistica.
    def enterRegLogistica(self, ctx:TintoParser.RegLogisticaContext):
        pass

    # Exit a parse tree produced by TintoParser#regLogistica.
    def exitRegLogistica(self, ctx:TintoParser.RegLogisticaContext):
        pass


    # Enter a parse tree produced by TintoParser#or.
    def enterOr(self, ctx:TintoParser.OrContext):
        pass

    # Exit a parse tree produced by TintoParser#or.
    def exitOr(self, ctx:TintoParser.OrContext):
        pass


    # Enter a parse tree produced by TintoParser#perceptron.
    def enterPerceptron(self, ctx:TintoParser.PerceptronContext):
        pass

    # Exit a parse tree produced by TintoParser#perceptron.
    def exitPerceptron(self, ctx:TintoParser.PerceptronContext):
        pass


    # Enter a parse tree produced by TintoParser#numero.
    def enterNumero(self, ctx:TintoParser.NumeroContext):
        pass

    # Exit a parse tree produced by TintoParser#numero.
    def exitNumero(self, ctx:TintoParser.NumeroContext):
        pass


    # Enter a parse tree produced by TintoParser#cadena.
    def enterCadena(self, ctx:TintoParser.CadenaContext):
        pass

    # Exit a parse tree produced by TintoParser#cadena.
    def exitCadena(self, ctx:TintoParser.CadenaContext):
        pass


    # Enter a parse tree produced by TintoParser#regLineal.
    def enterRegLineal(self, ctx:TintoParser.RegLinealContext):
        pass

    # Exit a parse tree produced by TintoParser#regLineal.
    def exitRegLineal(self, ctx:TintoParser.RegLinealContext):
        pass


    # Enter a parse tree produced by TintoParser#matriz.
    def enterMatriz(self, ctx:TintoParser.MatrizContext):
        pass

    # Exit a parse tree produced by TintoParser#matriz.
    def exitMatriz(self, ctx:TintoParser.MatrizContext):
        pass


    # Enter a parse tree produced by TintoParser#parentesis.
    def enterParentesis(self, ctx:TintoParser.ParentesisContext):
        pass

    # Exit a parse tree produced by TintoParser#parentesis.
    def exitParentesis(self, ctx:TintoParser.ParentesisContext):
        pass


    # Enter a parse tree produced by TintoParser#sumaResta.
    def enterSumaResta(self, ctx:TintoParser.SumaRestaContext):
        pass

    # Exit a parse tree produced by TintoParser#sumaResta.
    def exitSumaResta(self, ctx:TintoParser.SumaRestaContext):
        pass


    # Enter a parse tree produced by TintoParser#potencia.
    def enterPotencia(self, ctx:TintoParser.PotenciaContext):
        pass

    # Exit a parse tree produced by TintoParser#potencia.
    def exitPotencia(self, ctx:TintoParser.PotenciaContext):
        pass


    # Enter a parse tree produced by TintoParser#not.
    def enterNot(self, ctx:TintoParser.NotContext):
        pass

    # Exit a parse tree produced by TintoParser#not.
    def exitNot(self, ctx:TintoParser.NotContext):
        pass


    # Enter a parse tree produced by TintoParser#llamadaFuncion.
    def enterLlamadaFuncion(self, ctx:TintoParser.LlamadaFuncionContext):
        pass

    # Exit a parse tree produced by TintoParser#llamadaFuncion.
    def exitLlamadaFuncion(self, ctx:TintoParser.LlamadaFuncionContext):
        pass


    # Enter a parse tree produced by TintoParser#booleano.
    def enterBooleano(self, ctx:TintoParser.BooleanoContext):
        pass

    # Exit a parse tree produced by TintoParser#booleano.
    def exitBooleano(self, ctx:TintoParser.BooleanoContext):
        pass


    # Enter a parse tree produced by TintoParser#and.
    def enterAnd(self, ctx:TintoParser.AndContext):
        pass

    # Exit a parse tree produced by TintoParser#and.
    def exitAnd(self, ctx:TintoParser.AndContext):
        pass


    # Enter a parse tree produced by TintoParser#transpuesta.
    def enterTranspuesta(self, ctx:TintoParser.TranspuestaContext):
        pass

    # Exit a parse tree produced by TintoParser#transpuesta.
    def exitTranspuesta(self, ctx:TintoParser.TranspuestaContext):
        pass


    # Enter a parse tree produced by TintoParser#variable.
    def enterVariable(self, ctx:TintoParser.VariableContext):
        pass

    # Exit a parse tree produced by TintoParser#variable.
    def exitVariable(self, ctx:TintoParser.VariableContext):
        pass


    # Enter a parse tree produced by TintoParser#comparacion.
    def enterComparacion(self, ctx:TintoParser.ComparacionContext):
        pass

    # Exit a parse tree produced by TintoParser#comparacion.
    def exitComparacion(self, ctx:TintoParser.ComparacionContext):
        pass


    # Enter a parse tree produced by TintoParser#inversa.
    def enterInversa(self, ctx:TintoParser.InversaContext):
        pass

    # Exit a parse tree produced by TintoParser#inversa.
    def exitInversa(self, ctx:TintoParser.InversaContext):
        pass


    # Enter a parse tree produced by TintoParser#matrix.
    def enterMatrix(self, ctx:TintoParser.MatrixContext):
        pass

    # Exit a parse tree produced by TintoParser#matrix.
    def exitMatrix(self, ctx:TintoParser.MatrixContext):
        pass


    # Enter a parse tree produced by TintoParser#fila.
    def enterFila(self, ctx:TintoParser.FilaContext):
        pass

    # Exit a parse tree produced by TintoParser#fila.
    def exitFila(self, ctx:TintoParser.FilaContext):
        pass


    # Enter a parse tree produced by TintoParser#functionCall.
    def enterFunctionCall(self, ctx:TintoParser.FunctionCallContext):
        pass

    # Exit a parse tree produced by TintoParser#functionCall.
    def exitFunctionCall(self, ctx:TintoParser.FunctionCallContext):
        pass



del TintoParser