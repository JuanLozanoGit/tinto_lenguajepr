grammar Tinto;

program: statement+ EOF;

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

ifStatement: 'if' '(' expr ')' bloque ('else' bloque)? ;
whileStatement: 'while' '(' expr ')' bloque ;
forStatement: 'for' ID '=' expr 'to' expr bloque ;

bloque: '{' statement* '}' ;

returnStatement: 'return' expr ;
printStatement: 'print' '(' (expr (',' expr)*)? ')' ;
plotStatement: 'plot' '(' expr (',' expr)? ')' ;

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

NUMBER: [0-9]+ ('.' [0-9]+)? ;
BOOLEAN: 'true' | 'false' ;
STRING: '"' (~["\r\n])* '"' ;
ID: [a-zA-Z_][a-zA-Z0-9_]* ;
WS: [ \t\r\n]+ -> skip ;
COMMENT: '//' ~[\r\n]* -> skip ;
