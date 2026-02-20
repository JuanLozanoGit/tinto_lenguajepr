grammar Tinto;

// Regla inicial
program: statement+ EOF;

statement:
    variableDeclaration ';' 
    | functionCall ';'
    | ifStatement
    | whileStatement
    | forStatement
    | returnStatement ';'
    | assignment ';'
    | printStatement ';'
    | plotStatement ';' ; // Nuevo: Requisito de gráficas

// Expresiones con precedencia matemática
expression:
    <assoc=right> expression '^' expression      // Potencia x^y
    | expression ('*' | '/' | '%') expression
    | expression ('+' | '-') expression
    | '(' expression ')'
    | matrix                                     // Soporte para [[1,2],[3,4]]
    | functionCall
    | primary;

primary: NUMBER | ID | STRING | BOOLEAN;

// Operaciones de Matrices y DL (Funciones nativas)
matrix: '[' row (',' row)* ']';
row: '[' expression (',' expression)* ']';

// Funciones para DL
functionCall: ID '(' (expression (',' expression)*)? ')';
