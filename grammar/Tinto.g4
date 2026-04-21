grammar Tinto;

// --- Programa Principal ---
program: (functionDeclaration | statement)* EOF;

// --- Funciones ---
functionDeclaration: FUNC ID '(' parameters? ')' bloque;

parameters: parameter (',' parameter)*;
parameter : type ID;

// --- Sentencias ---
statement: variableDeclaration ';'
         | ifStatement
         | whileStatement
         | forStatement
         | returnStatement ';'
         | assignment ';'
         | printStatement ';'
         | plotStatement ';'
         | functionCall ';'
         | expr ';'
         ;

variableDeclaration: type? ID '=' expr ;
type: 'int' | 'float' | 'bool' | 'string' | 'matrix';
assignment: ID '=' expr ;

ifStatement: IF '(' expr ')' bloque (ELSE bloque)? ;
whileStatement: WHILE '(' expr ')' bloque ;
forStatement: FOR ID '=' expr TO expr bloque ;

bloque: '{' statement* '}' ;

returnStatement: RETURN expr ;
printStatement: PRINT '(' (expr (',' expr)*)? ')' ;
plotStatement: PLOT '(' expr (',' expr)? ')' ;

// --- Expresiones ---
expr: <assoc=right> expr '^' expr                # Potencia
    | expr op=('*'|'/'|'%') expr                 # MulDivMod
    | expr op=('+'|'-') expr                     # SumaResta
    | expr op=('=='|'!='|'<'|'>'|'<='|'>=') expr # Comparacion
    | '!' expr                                   # Not
    | expr '&&' expr                             # And
    | expr '||' expr                             # Or
    | '(' expr ')'                               # Parentesis
    | ('sin' | 'cos' | 'tan' | 'sqrt' | 'log') '(' expr ')' # FuncMat
    | 'transpose' '(' expr ')'                   # Transpuesta
    | 'inverse' '(' expr ')'                     # Inversa
    | 'linearRegression' '(' expr ',' expr ')'   # RegLineal
    | 'logisticRegression' '(' expr ',' expr ')' # RegLogistica
    | 'perceptron' '(' expr ',' expr ',' expr ')'# Perceptron
    | matrix                                     # Matriz
    | functionCall                               # LlamadaFuncion
    | NUMBER                                     # Numero
    | BOOLEAN                                    # Booleano
    | STRING                                     # Cadena
    | ID                                         # Variable
    ;

matrix: '[' fila (';' fila)* ']' ;
fila: expr (',' expr)* ;

functionCall: ID '(' (expr (',' expr)*)? ')' ;

// --- Tokens (Lexer) ---
FUNC   : 'func';
IF     : 'if';
ELSE   : 'else';
WHILE  : 'while';
FOR    : 'for';
TO     : 'to';
RETURN : 'return';
PRINT  : 'print';
PLOT   : 'plot';

NUMBER: [0-9]+ ('.' [0-9]+)? ;
BOOLEAN: 'true' | 'false' ;
STRING: '"' (~["\r\n])* '"' ;
ID: [a-zA-Z_][a-zA-Z0-9_]* ;

WS: [ \t\r\n]+ -> skip ;
COMMENT: '//' ~[\r\n]* -> skip ;
