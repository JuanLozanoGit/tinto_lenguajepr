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

expr: <assoc=right> expr '^' expr          # potencia
    | expr op=('*'|'/'|'%') expr           # mulDivMod
    | expr op=('+'|'-') expr               # sumaResta
    | expr op=('=='|'!='|'<'|'>'|'<='|'>=') expr # comparacion
    | '!' expr                              # not
    | expr '&&' expr                        # and
    | expr '||' expr                        # or
    | '(' expr ')'                          # parentesis
    | ('sin' | 'cos' | 'tan' | 'sqrt' | 'log') '(' expr ')' # funcMat
    | 'transpose' '(' expr ')'               # transpuesta
    | 'inverse' '(' expr ')'                  # inversa
    | 'linearRegression' '(' expr ',' expr ')' # regLineal
    | 'logisticRegression' '(' expr ',' expr ')' # regLogistica
    | 'perceptron' '(' expr ',' expr ',' expr ')' # perceptron
    | matrix                                 # matriz
    | functionCall                            # llamadaFuncion
    | NUMBER                                  # numero
    | BOOLEAN                                  # booleano
    | STRING                                   # cadena
    | ID                                       # variable
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
