# Generated from grammar/Tinto.g4 by ANTLR 4.13.1
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,53,246,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,1,0,4,0,34,8,0,11,0,12,0,35,1,0,1,0,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,3,1,64,8,1,1,2,3,2,67,8,2,1,2,1,2,1,2,1,
        2,1,3,1,3,1,4,1,4,1,4,1,4,1,5,1,5,1,5,1,5,1,5,1,5,1,5,3,5,86,8,5,
        1,6,1,6,1,6,1,6,1,6,1,6,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,7,1,8,1,8,
        5,8,104,8,8,10,8,12,8,107,9,8,1,8,1,8,1,9,1,9,1,9,1,10,1,10,1,10,
        1,10,1,10,5,10,119,8,10,10,10,12,10,122,9,10,3,10,124,8,10,1,10,
        1,10,1,11,1,11,1,11,1,11,1,11,3,11,133,8,11,1,11,1,11,1,12,1,12,
        1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,
        1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,
        1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,
        1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,3,12,188,8,12,
        1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,
        1,12,1,12,1,12,1,12,1,12,5,12,208,8,12,10,12,12,12,211,9,12,1,13,
        1,13,1,13,1,13,5,13,217,8,13,10,13,12,13,220,9,13,1,13,1,13,1,14,
        1,14,1,14,5,14,227,8,14,10,14,12,14,230,9,14,1,15,1,15,1,15,1,15,
        1,15,5,15,237,8,15,10,15,12,15,240,9,15,3,15,242,8,15,1,15,1,15,
        1,15,0,1,24,16,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,0,5,1,
        0,3,7,1,0,36,40,1,0,22,24,1,0,25,26,1,0,27,32,268,0,33,1,0,0,0,2,
        63,1,0,0,0,4,66,1,0,0,0,6,72,1,0,0,0,8,74,1,0,0,0,10,78,1,0,0,0,
        12,87,1,0,0,0,14,93,1,0,0,0,16,101,1,0,0,0,18,110,1,0,0,0,20,113,
        1,0,0,0,22,127,1,0,0,0,24,187,1,0,0,0,26,212,1,0,0,0,28,223,1,0,
        0,0,30,231,1,0,0,0,32,34,3,2,1,0,33,32,1,0,0,0,34,35,1,0,0,0,35,
        33,1,0,0,0,35,36,1,0,0,0,36,37,1,0,0,0,37,38,5,0,0,1,38,1,1,0,0,
        0,39,40,3,4,2,0,40,41,5,1,0,0,41,64,1,0,0,0,42,64,3,10,5,0,43,64,
        3,12,6,0,44,64,3,14,7,0,45,46,3,18,9,0,46,47,5,1,0,0,47,64,1,0,0,
        0,48,49,3,8,4,0,49,50,5,1,0,0,50,64,1,0,0,0,51,52,3,20,10,0,52,53,
        5,1,0,0,53,64,1,0,0,0,54,55,3,22,11,0,55,56,5,1,0,0,56,64,1,0,0,
        0,57,58,3,30,15,0,58,59,5,1,0,0,59,64,1,0,0,0,60,61,3,24,12,0,61,
        62,5,1,0,0,62,64,1,0,0,0,63,39,1,0,0,0,63,42,1,0,0,0,63,43,1,0,0,
        0,63,44,1,0,0,0,63,45,1,0,0,0,63,48,1,0,0,0,63,51,1,0,0,0,63,54,
        1,0,0,0,63,57,1,0,0,0,63,60,1,0,0,0,64,3,1,0,0,0,65,67,3,6,3,0,66,
        65,1,0,0,0,66,67,1,0,0,0,67,68,1,0,0,0,68,69,5,51,0,0,69,70,5,2,
        0,0,70,71,3,24,12,0,71,5,1,0,0,0,72,73,7,0,0,0,73,7,1,0,0,0,74,75,
        5,51,0,0,75,76,5,2,0,0,76,77,3,24,12,0,77,9,1,0,0,0,78,79,5,8,0,
        0,79,80,5,9,0,0,80,81,3,24,12,0,81,82,5,10,0,0,82,85,3,16,8,0,83,
        84,5,11,0,0,84,86,3,16,8,0,85,83,1,0,0,0,85,86,1,0,0,0,86,11,1,0,
        0,0,87,88,5,12,0,0,88,89,5,9,0,0,89,90,3,24,12,0,90,91,5,10,0,0,
        91,92,3,16,8,0,92,13,1,0,0,0,93,94,5,13,0,0,94,95,5,51,0,0,95,96,
        5,2,0,0,96,97,3,24,12,0,97,98,5,14,0,0,98,99,3,24,12,0,99,100,3,
        16,8,0,100,15,1,0,0,0,101,105,5,15,0,0,102,104,3,2,1,0,103,102,1,
        0,0,0,104,107,1,0,0,0,105,103,1,0,0,0,105,106,1,0,0,0,106,108,1,
        0,0,0,107,105,1,0,0,0,108,109,5,16,0,0,109,17,1,0,0,0,110,111,5,
        17,0,0,111,112,3,24,12,0,112,19,1,0,0,0,113,114,5,18,0,0,114,123,
        5,9,0,0,115,120,3,24,12,0,116,117,5,19,0,0,117,119,3,24,12,0,118,
        116,1,0,0,0,119,122,1,0,0,0,120,118,1,0,0,0,120,121,1,0,0,0,121,
        124,1,0,0,0,122,120,1,0,0,0,123,115,1,0,0,0,123,124,1,0,0,0,124,
        125,1,0,0,0,125,126,5,10,0,0,126,21,1,0,0,0,127,128,5,20,0,0,128,
        129,5,9,0,0,129,132,3,24,12,0,130,131,5,19,0,0,131,133,3,24,12,0,
        132,130,1,0,0,0,132,133,1,0,0,0,133,134,1,0,0,0,134,135,5,10,0,0,
        135,23,1,0,0,0,136,137,6,12,-1,0,137,138,5,33,0,0,138,188,3,24,12,
        16,139,140,5,9,0,0,140,141,3,24,12,0,141,142,5,10,0,0,142,188,1,
        0,0,0,143,144,7,1,0,0,144,145,5,9,0,0,145,146,3,24,12,0,146,147,
        5,10,0,0,147,188,1,0,0,0,148,149,5,41,0,0,149,150,5,9,0,0,150,151,
        3,24,12,0,151,152,5,10,0,0,152,188,1,0,0,0,153,154,5,42,0,0,154,
        155,5,9,0,0,155,156,3,24,12,0,156,157,5,10,0,0,157,188,1,0,0,0,158,
        159,5,43,0,0,159,160,5,9,0,0,160,161,3,24,12,0,161,162,5,19,0,0,
        162,163,3,24,12,0,163,164,5,10,0,0,164,188,1,0,0,0,165,166,5,44,
        0,0,166,167,5,9,0,0,167,168,3,24,12,0,168,169,5,19,0,0,169,170,3,
        24,12,0,170,171,5,10,0,0,171,188,1,0,0,0,172,173,5,45,0,0,173,174,
        5,9,0,0,174,175,3,24,12,0,175,176,5,19,0,0,176,177,3,24,12,0,177,
        178,5,19,0,0,178,179,3,24,12,0,179,180,5,10,0,0,180,188,1,0,0,0,
        181,188,3,26,13,0,182,188,3,30,15,0,183,188,5,48,0,0,184,188,5,49,
        0,0,185,188,5,50,0,0,186,188,5,51,0,0,187,136,1,0,0,0,187,139,1,
        0,0,0,187,143,1,0,0,0,187,148,1,0,0,0,187,153,1,0,0,0,187,158,1,
        0,0,0,187,165,1,0,0,0,187,172,1,0,0,0,187,181,1,0,0,0,187,182,1,
        0,0,0,187,183,1,0,0,0,187,184,1,0,0,0,187,185,1,0,0,0,187,186,1,
        0,0,0,188,209,1,0,0,0,189,190,10,20,0,0,190,191,5,21,0,0,191,208,
        3,24,12,20,192,193,10,19,0,0,193,194,7,2,0,0,194,208,3,24,12,20,
        195,196,10,18,0,0,196,197,7,3,0,0,197,208,3,24,12,19,198,199,10,
        17,0,0,199,200,7,4,0,0,200,208,3,24,12,18,201,202,10,15,0,0,202,
        203,5,34,0,0,203,208,3,24,12,16,204,205,10,14,0,0,205,206,5,35,0,
        0,206,208,3,24,12,15,207,189,1,0,0,0,207,192,1,0,0,0,207,195,1,0,
        0,0,207,198,1,0,0,0,207,201,1,0,0,0,207,204,1,0,0,0,208,211,1,0,
        0,0,209,207,1,0,0,0,209,210,1,0,0,0,210,25,1,0,0,0,211,209,1,0,0,
        0,212,213,5,46,0,0,213,218,3,28,14,0,214,215,5,1,0,0,215,217,3,28,
        14,0,216,214,1,0,0,0,217,220,1,0,0,0,218,216,1,0,0,0,218,219,1,0,
        0,0,219,221,1,0,0,0,220,218,1,0,0,0,221,222,5,47,0,0,222,27,1,0,
        0,0,223,228,3,24,12,0,224,225,5,19,0,0,225,227,3,24,12,0,226,224,
        1,0,0,0,227,230,1,0,0,0,228,226,1,0,0,0,228,229,1,0,0,0,229,29,1,
        0,0,0,230,228,1,0,0,0,231,232,5,51,0,0,232,241,5,9,0,0,233,238,3,
        24,12,0,234,235,5,19,0,0,235,237,3,24,12,0,236,234,1,0,0,0,237,240,
        1,0,0,0,238,236,1,0,0,0,238,239,1,0,0,0,239,242,1,0,0,0,240,238,
        1,0,0,0,241,233,1,0,0,0,241,242,1,0,0,0,242,243,1,0,0,0,243,244,
        5,10,0,0,244,31,1,0,0,0,15,35,63,66,85,105,120,123,132,187,207,209,
        218,228,238,241
    ]

class TintoParser ( Parser ):

    grammarFileName = "Tinto.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "';'", "'='", "'int'", "'float'", "'bool'", 
                     "'string'", "'matrix'", "'if'", "'('", "')'", "'else'", 
                     "'while'", "'for'", "'to'", "'{'", "'}'", "'return'", 
                     "'print'", "','", "'plot'", "'^'", "'*'", "'/'", "'%'", 
                     "'+'", "'-'", "'=='", "'!='", "'<'", "'>'", "'<='", 
                     "'>='", "'!'", "'&&'", "'||'", "'sin'", "'cos'", "'tan'", 
                     "'sqrt'", "'log'", "'transpose'", "'inverse'", "'linearRegression'", 
                     "'logisticRegression'", "'perceptron'", "'['", "']'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "NUMBER", "BOOLEAN", "STRING", "ID", "WS", "COMMENT" ]

    RULE_program = 0
    RULE_statement = 1
    RULE_variableDeclaration = 2
    RULE_type = 3
    RULE_assignment = 4
    RULE_ifStatement = 5
    RULE_whileStatement = 6
    RULE_forStatement = 7
    RULE_bloque = 8
    RULE_returnStatement = 9
    RULE_printStatement = 10
    RULE_plotStatement = 11
    RULE_expr = 12
    RULE_matrix = 13
    RULE_fila = 14
    RULE_functionCall = 15

    ruleNames =  [ "program", "statement", "variableDeclaration", "type", 
                   "assignment", "ifStatement", "whileStatement", "forStatement", 
                   "bloque", "returnStatement", "printStatement", "plotStatement", 
                   "expr", "matrix", "fila", "functionCall" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    T__3=4
    T__4=5
    T__5=6
    T__6=7
    T__7=8
    T__8=9
    T__9=10
    T__10=11
    T__11=12
    T__12=13
    T__13=14
    T__14=15
    T__15=16
    T__16=17
    T__17=18
    T__18=19
    T__19=20
    T__20=21
    T__21=22
    T__22=23
    T__23=24
    T__24=25
    T__25=26
    T__26=27
    T__27=28
    T__28=29
    T__29=30
    T__30=31
    T__31=32
    T__32=33
    T__33=34
    T__34=35
    T__35=36
    T__36=37
    T__37=38
    T__38=39
    T__39=40
    T__40=41
    T__41=42
    T__42=43
    T__43=44
    T__44=45
    T__45=46
    T__46=47
    NUMBER=48
    BOOLEAN=49
    STRING=50
    ID=51
    WS=52
    COMMENT=53

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.13.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class ProgramContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EOF(self):
            return self.getToken(TintoParser.EOF, 0)

        def statement(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TintoParser.StatementContext)
            else:
                return self.getTypedRuleContext(TintoParser.StatementContext,i)


        def getRuleIndex(self):
            return TintoParser.RULE_program

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterProgram" ):
                listener.enterProgram(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitProgram" ):
                listener.exitProgram(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitProgram" ):
                return visitor.visitProgram(self)
            else:
                return visitor.visitChildren(self)




    def program(self):

        localctx = TintoParser.ProgramContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_program)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 33 
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 32
                self.statement()
                self.state = 35 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not ((((_la) & ~0x3f) == 0 and ((1 << _la) & 4362802010928120) != 0)):
                    break

            self.state = 37
            self.match(TintoParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class StatementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def variableDeclaration(self):
            return self.getTypedRuleContext(TintoParser.VariableDeclarationContext,0)


        def ifStatement(self):
            return self.getTypedRuleContext(TintoParser.IfStatementContext,0)


        def whileStatement(self):
            return self.getTypedRuleContext(TintoParser.WhileStatementContext,0)


        def forStatement(self):
            return self.getTypedRuleContext(TintoParser.ForStatementContext,0)


        def returnStatement(self):
            return self.getTypedRuleContext(TintoParser.ReturnStatementContext,0)


        def assignment(self):
            return self.getTypedRuleContext(TintoParser.AssignmentContext,0)


        def printStatement(self):
            return self.getTypedRuleContext(TintoParser.PrintStatementContext,0)


        def plotStatement(self):
            return self.getTypedRuleContext(TintoParser.PlotStatementContext,0)


        def functionCall(self):
            return self.getTypedRuleContext(TintoParser.FunctionCallContext,0)


        def expr(self):
            return self.getTypedRuleContext(TintoParser.ExprContext,0)


        def getRuleIndex(self):
            return TintoParser.RULE_statement

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterStatement" ):
                listener.enterStatement(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitStatement" ):
                listener.exitStatement(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitStatement" ):
                return visitor.visitStatement(self)
            else:
                return visitor.visitChildren(self)




    def statement(self):

        localctx = TintoParser.StatementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_statement)
        try:
            self.state = 63
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,1,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 39
                self.variableDeclaration()
                self.state = 40
                self.match(TintoParser.T__0)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 42
                self.ifStatement()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 43
                self.whileStatement()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 44
                self.forStatement()
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 45
                self.returnStatement()
                self.state = 46
                self.match(TintoParser.T__0)
                pass

            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 48
                self.assignment()
                self.state = 49
                self.match(TintoParser.T__0)
                pass

            elif la_ == 7:
                self.enterOuterAlt(localctx, 7)
                self.state = 51
                self.printStatement()
                self.state = 52
                self.match(TintoParser.T__0)
                pass

            elif la_ == 8:
                self.enterOuterAlt(localctx, 8)
                self.state = 54
                self.plotStatement()
                self.state = 55
                self.match(TintoParser.T__0)
                pass

            elif la_ == 9:
                self.enterOuterAlt(localctx, 9)
                self.state = 57
                self.functionCall()
                self.state = 58
                self.match(TintoParser.T__0)
                pass

            elif la_ == 10:
                self.enterOuterAlt(localctx, 10)
                self.state = 60
                self.expr(0)
                self.state = 61
                self.match(TintoParser.T__0)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class VariableDeclarationContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(TintoParser.ID, 0)

        def expr(self):
            return self.getTypedRuleContext(TintoParser.ExprContext,0)


        def type_(self):
            return self.getTypedRuleContext(TintoParser.TypeContext,0)


        def getRuleIndex(self):
            return TintoParser.RULE_variableDeclaration

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterVariableDeclaration" ):
                listener.enterVariableDeclaration(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitVariableDeclaration" ):
                listener.exitVariableDeclaration(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitVariableDeclaration" ):
                return visitor.visitVariableDeclaration(self)
            else:
                return visitor.visitChildren(self)




    def variableDeclaration(self):

        localctx = TintoParser.VariableDeclarationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_variableDeclaration)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 66
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if (((_la) & ~0x3f) == 0 and ((1 << _la) & 248) != 0):
                self.state = 65
                self.type_()


            self.state = 68
            self.match(TintoParser.ID)
            self.state = 69
            self.match(TintoParser.T__1)
            self.state = 70
            self.expr(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TypeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return TintoParser.RULE_type

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterType" ):
                listener.enterType(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitType" ):
                listener.exitType(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitType" ):
                return visitor.visitType(self)
            else:
                return visitor.visitChildren(self)




    def type_(self):

        localctx = TintoParser.TypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_type)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 72
            _la = self._input.LA(1)
            if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 248) != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AssignmentContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(TintoParser.ID, 0)

        def expr(self):
            return self.getTypedRuleContext(TintoParser.ExprContext,0)


        def getRuleIndex(self):
            return TintoParser.RULE_assignment

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAssignment" ):
                listener.enterAssignment(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAssignment" ):
                listener.exitAssignment(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAssignment" ):
                return visitor.visitAssignment(self)
            else:
                return visitor.visitChildren(self)




    def assignment(self):

        localctx = TintoParser.AssignmentContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_assignment)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 74
            self.match(TintoParser.ID)
            self.state = 75
            self.match(TintoParser.T__1)
            self.state = 76
            self.expr(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class IfStatementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self):
            return self.getTypedRuleContext(TintoParser.ExprContext,0)


        def bloque(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TintoParser.BloqueContext)
            else:
                return self.getTypedRuleContext(TintoParser.BloqueContext,i)


        def getRuleIndex(self):
            return TintoParser.RULE_ifStatement

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterIfStatement" ):
                listener.enterIfStatement(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitIfStatement" ):
                listener.exitIfStatement(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitIfStatement" ):
                return visitor.visitIfStatement(self)
            else:
                return visitor.visitChildren(self)




    def ifStatement(self):

        localctx = TintoParser.IfStatementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_ifStatement)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 78
            self.match(TintoParser.T__7)
            self.state = 79
            self.match(TintoParser.T__8)
            self.state = 80
            self.expr(0)
            self.state = 81
            self.match(TintoParser.T__9)
            self.state = 82
            self.bloque()
            self.state = 85
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==11:
                self.state = 83
                self.match(TintoParser.T__10)
                self.state = 84
                self.bloque()


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class WhileStatementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self):
            return self.getTypedRuleContext(TintoParser.ExprContext,0)


        def bloque(self):
            return self.getTypedRuleContext(TintoParser.BloqueContext,0)


        def getRuleIndex(self):
            return TintoParser.RULE_whileStatement

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterWhileStatement" ):
                listener.enterWhileStatement(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitWhileStatement" ):
                listener.exitWhileStatement(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitWhileStatement" ):
                return visitor.visitWhileStatement(self)
            else:
                return visitor.visitChildren(self)




    def whileStatement(self):

        localctx = TintoParser.WhileStatementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_whileStatement)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 87
            self.match(TintoParser.T__11)
            self.state = 88
            self.match(TintoParser.T__8)
            self.state = 89
            self.expr(0)
            self.state = 90
            self.match(TintoParser.T__9)
            self.state = 91
            self.bloque()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ForStatementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(TintoParser.ID, 0)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TintoParser.ExprContext)
            else:
                return self.getTypedRuleContext(TintoParser.ExprContext,i)


        def bloque(self):
            return self.getTypedRuleContext(TintoParser.BloqueContext,0)


        def getRuleIndex(self):
            return TintoParser.RULE_forStatement

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterForStatement" ):
                listener.enterForStatement(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitForStatement" ):
                listener.exitForStatement(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitForStatement" ):
                return visitor.visitForStatement(self)
            else:
                return visitor.visitChildren(self)




    def forStatement(self):

        localctx = TintoParser.ForStatementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_forStatement)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 93
            self.match(TintoParser.T__12)
            self.state = 94
            self.match(TintoParser.ID)
            self.state = 95
            self.match(TintoParser.T__1)
            self.state = 96
            self.expr(0)
            self.state = 97
            self.match(TintoParser.T__13)
            self.state = 98
            self.expr(0)
            self.state = 99
            self.bloque()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class BloqueContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def statement(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TintoParser.StatementContext)
            else:
                return self.getTypedRuleContext(TintoParser.StatementContext,i)


        def getRuleIndex(self):
            return TintoParser.RULE_bloque

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBloque" ):
                listener.enterBloque(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBloque" ):
                listener.exitBloque(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBloque" ):
                return visitor.visitBloque(self)
            else:
                return visitor.visitChildren(self)




    def bloque(self):

        localctx = TintoParser.BloqueContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_bloque)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 101
            self.match(TintoParser.T__14)
            self.state = 105
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while (((_la) & ~0x3f) == 0 and ((1 << _la) & 4362802010928120) != 0):
                self.state = 102
                self.statement()
                self.state = 107
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 108
            self.match(TintoParser.T__15)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ReturnStatementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self):
            return self.getTypedRuleContext(TintoParser.ExprContext,0)


        def getRuleIndex(self):
            return TintoParser.RULE_returnStatement

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterReturnStatement" ):
                listener.enterReturnStatement(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitReturnStatement" ):
                listener.exitReturnStatement(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitReturnStatement" ):
                return visitor.visitReturnStatement(self)
            else:
                return visitor.visitChildren(self)




    def returnStatement(self):

        localctx = TintoParser.ReturnStatementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_returnStatement)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 110
            self.match(TintoParser.T__16)
            self.state = 111
            self.expr(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PrintStatementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TintoParser.ExprContext)
            else:
                return self.getTypedRuleContext(TintoParser.ExprContext,i)


        def getRuleIndex(self):
            return TintoParser.RULE_printStatement

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPrintStatement" ):
                listener.enterPrintStatement(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPrintStatement" ):
                listener.exitPrintStatement(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPrintStatement" ):
                return visitor.visitPrintStatement(self)
            else:
                return visitor.visitChildren(self)




    def printStatement(self):

        localctx = TintoParser.PrintStatementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_printStatement)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 113
            self.match(TintoParser.T__17)
            self.state = 114
            self.match(TintoParser.T__8)
            self.state = 123
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if (((_la) & ~0x3f) == 0 and ((1 << _la) & 4362802009473536) != 0):
                self.state = 115
                self.expr(0)
                self.state = 120
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la==19:
                    self.state = 116
                    self.match(TintoParser.T__18)
                    self.state = 117
                    self.expr(0)
                    self.state = 122
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)



            self.state = 125
            self.match(TintoParser.T__9)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PlotStatementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TintoParser.ExprContext)
            else:
                return self.getTypedRuleContext(TintoParser.ExprContext,i)


        def getRuleIndex(self):
            return TintoParser.RULE_plotStatement

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPlotStatement" ):
                listener.enterPlotStatement(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPlotStatement" ):
                listener.exitPlotStatement(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPlotStatement" ):
                return visitor.visitPlotStatement(self)
            else:
                return visitor.visitChildren(self)




    def plotStatement(self):

        localctx = TintoParser.PlotStatementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_plotStatement)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 127
            self.match(TintoParser.T__19)
            self.state = 128
            self.match(TintoParser.T__8)
            self.state = 129
            self.expr(0)
            self.state = 132
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==19:
                self.state = 130
                self.match(TintoParser.T__18)
                self.state = 131
                self.expr(0)


            self.state = 134
            self.match(TintoParser.T__9)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ExprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return TintoParser.RULE_expr

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)


    class MulDivModContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a TintoParser.ExprContext
            super().__init__(parser)
            self.op = None # Token
            self.copyFrom(ctx)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TintoParser.ExprContext)
            else:
                return self.getTypedRuleContext(TintoParser.ExprContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMulDivMod" ):
                listener.enterMulDivMod(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMulDivMod" ):
                listener.exitMulDivMod(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMulDivMod" ):
                return visitor.visitMulDivMod(self)
            else:
                return visitor.visitChildren(self)


    class FuncMatContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a TintoParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self):
            return self.getTypedRuleContext(TintoParser.ExprContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFuncMat" ):
                listener.enterFuncMat(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFuncMat" ):
                listener.exitFuncMat(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFuncMat" ):
                return visitor.visitFuncMat(self)
            else:
                return visitor.visitChildren(self)


    class RegLogisticaContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a TintoParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TintoParser.ExprContext)
            else:
                return self.getTypedRuleContext(TintoParser.ExprContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterRegLogistica" ):
                listener.enterRegLogistica(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitRegLogistica" ):
                listener.exitRegLogistica(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitRegLogistica" ):
                return visitor.visitRegLogistica(self)
            else:
                return visitor.visitChildren(self)


    class OrContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a TintoParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TintoParser.ExprContext)
            else:
                return self.getTypedRuleContext(TintoParser.ExprContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterOr" ):
                listener.enterOr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitOr" ):
                listener.exitOr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitOr" ):
                return visitor.visitOr(self)
            else:
                return visitor.visitChildren(self)


    class PerceptronContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a TintoParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TintoParser.ExprContext)
            else:
                return self.getTypedRuleContext(TintoParser.ExprContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPerceptron" ):
                listener.enterPerceptron(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPerceptron" ):
                listener.exitPerceptron(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPerceptron" ):
                return visitor.visitPerceptron(self)
            else:
                return visitor.visitChildren(self)


    class NumeroContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a TintoParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def NUMBER(self):
            return self.getToken(TintoParser.NUMBER, 0)

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterNumero" ):
                listener.enterNumero(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitNumero" ):
                listener.exitNumero(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitNumero" ):
                return visitor.visitNumero(self)
            else:
                return visitor.visitChildren(self)


    class CadenaContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a TintoParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def STRING(self):
            return self.getToken(TintoParser.STRING, 0)

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterCadena" ):
                listener.enterCadena(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitCadena" ):
                listener.exitCadena(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitCadena" ):
                return visitor.visitCadena(self)
            else:
                return visitor.visitChildren(self)


    class RegLinealContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a TintoParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TintoParser.ExprContext)
            else:
                return self.getTypedRuleContext(TintoParser.ExprContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterRegLineal" ):
                listener.enterRegLineal(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitRegLineal" ):
                listener.exitRegLineal(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitRegLineal" ):
                return visitor.visitRegLineal(self)
            else:
                return visitor.visitChildren(self)


    class MatrizContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a TintoParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def matrix(self):
            return self.getTypedRuleContext(TintoParser.MatrixContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMatriz" ):
                listener.enterMatriz(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMatriz" ):
                listener.exitMatriz(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMatriz" ):
                return visitor.visitMatriz(self)
            else:
                return visitor.visitChildren(self)


    class ParentesisContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a TintoParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self):
            return self.getTypedRuleContext(TintoParser.ExprContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterParentesis" ):
                listener.enterParentesis(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitParentesis" ):
                listener.exitParentesis(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitParentesis" ):
                return visitor.visitParentesis(self)
            else:
                return visitor.visitChildren(self)


    class SumaRestaContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a TintoParser.ExprContext
            super().__init__(parser)
            self.op = None # Token
            self.copyFrom(ctx)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TintoParser.ExprContext)
            else:
                return self.getTypedRuleContext(TintoParser.ExprContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSumaResta" ):
                listener.enterSumaResta(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSumaResta" ):
                listener.exitSumaResta(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSumaResta" ):
                return visitor.visitSumaResta(self)
            else:
                return visitor.visitChildren(self)


    class PotenciaContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a TintoParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TintoParser.ExprContext)
            else:
                return self.getTypedRuleContext(TintoParser.ExprContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPotencia" ):
                listener.enterPotencia(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPotencia" ):
                listener.exitPotencia(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPotencia" ):
                return visitor.visitPotencia(self)
            else:
                return visitor.visitChildren(self)


    class NotContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a TintoParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self):
            return self.getTypedRuleContext(TintoParser.ExprContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterNot" ):
                listener.enterNot(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitNot" ):
                listener.exitNot(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitNot" ):
                return visitor.visitNot(self)
            else:
                return visitor.visitChildren(self)


    class LlamadaFuncionContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a TintoParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def functionCall(self):
            return self.getTypedRuleContext(TintoParser.FunctionCallContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLlamadaFuncion" ):
                listener.enterLlamadaFuncion(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLlamadaFuncion" ):
                listener.exitLlamadaFuncion(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitLlamadaFuncion" ):
                return visitor.visitLlamadaFuncion(self)
            else:
                return visitor.visitChildren(self)


    class BooleanoContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a TintoParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def BOOLEAN(self):
            return self.getToken(TintoParser.BOOLEAN, 0)

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBooleano" ):
                listener.enterBooleano(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBooleano" ):
                listener.exitBooleano(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBooleano" ):
                return visitor.visitBooleano(self)
            else:
                return visitor.visitChildren(self)


    class AndContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a TintoParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TintoParser.ExprContext)
            else:
                return self.getTypedRuleContext(TintoParser.ExprContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAnd" ):
                listener.enterAnd(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAnd" ):
                listener.exitAnd(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAnd" ):
                return visitor.visitAnd(self)
            else:
                return visitor.visitChildren(self)


    class TranspuestaContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a TintoParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self):
            return self.getTypedRuleContext(TintoParser.ExprContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTranspuesta" ):
                listener.enterTranspuesta(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTranspuesta" ):
                listener.exitTranspuesta(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTranspuesta" ):
                return visitor.visitTranspuesta(self)
            else:
                return visitor.visitChildren(self)


    class VariableContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a TintoParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def ID(self):
            return self.getToken(TintoParser.ID, 0)

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterVariable" ):
                listener.enterVariable(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitVariable" ):
                listener.exitVariable(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitVariable" ):
                return visitor.visitVariable(self)
            else:
                return visitor.visitChildren(self)


    class ComparacionContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a TintoParser.ExprContext
            super().__init__(parser)
            self.op = None # Token
            self.copyFrom(ctx)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TintoParser.ExprContext)
            else:
                return self.getTypedRuleContext(TintoParser.ExprContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterComparacion" ):
                listener.enterComparacion(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitComparacion" ):
                listener.exitComparacion(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitComparacion" ):
                return visitor.visitComparacion(self)
            else:
                return visitor.visitChildren(self)


    class InversaContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a TintoParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self):
            return self.getTypedRuleContext(TintoParser.ExprContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterInversa" ):
                listener.enterInversa(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitInversa" ):
                listener.exitInversa(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitInversa" ):
                return visitor.visitInversa(self)
            else:
                return visitor.visitChildren(self)



    def expr(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = TintoParser.ExprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 24
        self.enterRecursionRule(localctx, 24, self.RULE_expr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 187
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,8,self._ctx)
            if la_ == 1:
                localctx = TintoParser.NotContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx

                self.state = 137
                self.match(TintoParser.T__32)
                self.state = 138
                self.expr(16)
                pass

            elif la_ == 2:
                localctx = TintoParser.ParentesisContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 139
                self.match(TintoParser.T__8)
                self.state = 140
                self.expr(0)
                self.state = 141
                self.match(TintoParser.T__9)
                pass

            elif la_ == 3:
                localctx = TintoParser.FuncMatContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 143
                _la = self._input.LA(1)
                if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 2130303778816) != 0)):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 144
                self.match(TintoParser.T__8)
                self.state = 145
                self.expr(0)
                self.state = 146
                self.match(TintoParser.T__9)
                pass

            elif la_ == 4:
                localctx = TintoParser.TranspuestaContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 148
                self.match(TintoParser.T__40)
                self.state = 149
                self.match(TintoParser.T__8)
                self.state = 150
                self.expr(0)
                self.state = 151
                self.match(TintoParser.T__9)
                pass

            elif la_ == 5:
                localctx = TintoParser.InversaContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 153
                self.match(TintoParser.T__41)
                self.state = 154
                self.match(TintoParser.T__8)
                self.state = 155
                self.expr(0)
                self.state = 156
                self.match(TintoParser.T__9)
                pass

            elif la_ == 6:
                localctx = TintoParser.RegLinealContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 158
                self.match(TintoParser.T__42)
                self.state = 159
                self.match(TintoParser.T__8)
                self.state = 160
                self.expr(0)
                self.state = 161
                self.match(TintoParser.T__18)
                self.state = 162
                self.expr(0)
                self.state = 163
                self.match(TintoParser.T__9)
                pass

            elif la_ == 7:
                localctx = TintoParser.RegLogisticaContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 165
                self.match(TintoParser.T__43)
                self.state = 166
                self.match(TintoParser.T__8)
                self.state = 167
                self.expr(0)
                self.state = 168
                self.match(TintoParser.T__18)
                self.state = 169
                self.expr(0)
                self.state = 170
                self.match(TintoParser.T__9)
                pass

            elif la_ == 8:
                localctx = TintoParser.PerceptronContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 172
                self.match(TintoParser.T__44)
                self.state = 173
                self.match(TintoParser.T__8)
                self.state = 174
                self.expr(0)
                self.state = 175
                self.match(TintoParser.T__18)
                self.state = 176
                self.expr(0)
                self.state = 177
                self.match(TintoParser.T__18)
                self.state = 178
                self.expr(0)
                self.state = 179
                self.match(TintoParser.T__9)
                pass

            elif la_ == 9:
                localctx = TintoParser.MatrizContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 181
                self.matrix()
                pass

            elif la_ == 10:
                localctx = TintoParser.LlamadaFuncionContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 182
                self.functionCall()
                pass

            elif la_ == 11:
                localctx = TintoParser.NumeroContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 183
                self.match(TintoParser.NUMBER)
                pass

            elif la_ == 12:
                localctx = TintoParser.BooleanoContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 184
                self.match(TintoParser.BOOLEAN)
                pass

            elif la_ == 13:
                localctx = TintoParser.CadenaContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 185
                self.match(TintoParser.STRING)
                pass

            elif la_ == 14:
                localctx = TintoParser.VariableContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 186
                self.match(TintoParser.ID)
                pass


            self._ctx.stop = self._input.LT(-1)
            self.state = 209
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,10,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 207
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,9,self._ctx)
                    if la_ == 1:
                        localctx = TintoParser.PotenciaContext(self, TintoParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 189
                        if not self.precpred(self._ctx, 20):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 20)")
                        self.state = 190
                        self.match(TintoParser.T__20)
                        self.state = 191
                        self.expr(20)
                        pass

                    elif la_ == 2:
                        localctx = TintoParser.MulDivModContext(self, TintoParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 192
                        if not self.precpred(self._ctx, 19):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 19)")
                        self.state = 193
                        localctx.op = self._input.LT(1)
                        _la = self._input.LA(1)
                        if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 29360128) != 0)):
                            localctx.op = self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 194
                        self.expr(20)
                        pass

                    elif la_ == 3:
                        localctx = TintoParser.SumaRestaContext(self, TintoParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 195
                        if not self.precpred(self._ctx, 18):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 18)")
                        self.state = 196
                        localctx.op = self._input.LT(1)
                        _la = self._input.LA(1)
                        if not(_la==25 or _la==26):
                            localctx.op = self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 197
                        self.expr(19)
                        pass

                    elif la_ == 4:
                        localctx = TintoParser.ComparacionContext(self, TintoParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 198
                        if not self.precpred(self._ctx, 17):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 17)")
                        self.state = 199
                        localctx.op = self._input.LT(1)
                        _la = self._input.LA(1)
                        if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 8455716864) != 0)):
                            localctx.op = self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 200
                        self.expr(18)
                        pass

                    elif la_ == 5:
                        localctx = TintoParser.AndContext(self, TintoParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 201
                        if not self.precpred(self._ctx, 15):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 15)")
                        self.state = 202
                        self.match(TintoParser.T__33)
                        self.state = 203
                        self.expr(16)
                        pass

                    elif la_ == 6:
                        localctx = TintoParser.OrContext(self, TintoParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 204
                        if not self.precpred(self._ctx, 14):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 14)")
                        self.state = 205
                        self.match(TintoParser.T__34)
                        self.state = 206
                        self.expr(15)
                        pass

             
                self.state = 211
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,10,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class MatrixContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fila(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TintoParser.FilaContext)
            else:
                return self.getTypedRuleContext(TintoParser.FilaContext,i)


        def getRuleIndex(self):
            return TintoParser.RULE_matrix

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMatrix" ):
                listener.enterMatrix(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMatrix" ):
                listener.exitMatrix(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMatrix" ):
                return visitor.visitMatrix(self)
            else:
                return visitor.visitChildren(self)




    def matrix(self):

        localctx = TintoParser.MatrixContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_matrix)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 212
            self.match(TintoParser.T__45)
            self.state = 213
            self.fila()
            self.state = 218
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==1:
                self.state = 214
                self.match(TintoParser.T__0)
                self.state = 215
                self.fila()
                self.state = 220
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 221
            self.match(TintoParser.T__46)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class FilaContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TintoParser.ExprContext)
            else:
                return self.getTypedRuleContext(TintoParser.ExprContext,i)


        def getRuleIndex(self):
            return TintoParser.RULE_fila

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFila" ):
                listener.enterFila(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFila" ):
                listener.exitFila(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFila" ):
                return visitor.visitFila(self)
            else:
                return visitor.visitChildren(self)




    def fila(self):

        localctx = TintoParser.FilaContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_fila)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 223
            self.expr(0)
            self.state = 228
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==19:
                self.state = 224
                self.match(TintoParser.T__18)
                self.state = 225
                self.expr(0)
                self.state = 230
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class FunctionCallContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(TintoParser.ID, 0)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(TintoParser.ExprContext)
            else:
                return self.getTypedRuleContext(TintoParser.ExprContext,i)


        def getRuleIndex(self):
            return TintoParser.RULE_functionCall

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFunctionCall" ):
                listener.enterFunctionCall(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFunctionCall" ):
                listener.exitFunctionCall(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFunctionCall" ):
                return visitor.visitFunctionCall(self)
            else:
                return visitor.visitChildren(self)




    def functionCall(self):

        localctx = TintoParser.FunctionCallContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_functionCall)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 231
            self.match(TintoParser.ID)
            self.state = 232
            self.match(TintoParser.T__8)
            self.state = 241
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if (((_la) & ~0x3f) == 0 and ((1 << _la) & 4362802009473536) != 0):
                self.state = 233
                self.expr(0)
                self.state = 238
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la==19:
                    self.state = 234
                    self.match(TintoParser.T__18)
                    self.state = 235
                    self.expr(0)
                    self.state = 240
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)



            self.state = 243
            self.match(TintoParser.T__9)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx



    def sempred(self, localctx:RuleContext, ruleIndex:int, predIndex:int):
        if self._predicates == None:
            self._predicates = dict()
        self._predicates[12] = self.expr_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception("No predicate with index:" + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def expr_sempred(self, localctx:ExprContext, predIndex:int):
            if predIndex == 0:
                return self.precpred(self._ctx, 20)
         

            if predIndex == 1:
                return self.precpred(self._ctx, 19)
         

            if predIndex == 2:
                return self.precpred(self._ctx, 18)
         

            if predIndex == 3:
                return self.precpred(self._ctx, 17)
         

            if predIndex == 4:
                return self.precpred(self._ctx, 15)
         

            if predIndex == 5:
                return self.precpred(self._ctx, 14)
         




