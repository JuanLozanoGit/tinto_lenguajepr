grammar Tinto;

program: statement+ EOF;

statement: functionDeclaration
         | variableDeclaration ';'
         | ifStatement
         | whileStatement
         | forStatement
         | returnStatement ';'
         | assignment ';'
         | assignmentIndexed ';'
         | printStatement ';'
         | plotStatement ';'
         | functionCall ';'
         | expr ';'
         ;

variableDeclaration: tipo? ID '=' expr ;
tipo: 'int' | 'float' | 'bool' | 'string' | 'matrix' | 'stack';
assignment: ID '=' expr ;
assignmentIndexed: ID '[' expr ']' '=' expr ;

ifStatement: 'if' '(' expr ')' bloque ('else' bloque)? ;
whileStatement: 'while' '(' expr ')' bloque ;
forStatement: 'for' ID '=' expr 'to' expr bloque ;

bloque: '{' statement* '}' ;

returnStatement: 'return' expr ;
printStatement: 'print' '(' (expr (',' expr)*)? ')' ;
plotStatement: 'plot' '(' expr (',' expr)? ')' ;

functionDeclaration
    : FUNC ID '(' parameters? ')' bloque
    ;

parameters
    : ID (',' ID)*
    ;

// ------------------------
expr: <assoc=right> expr '^' expr                # Potencia
    | expr '[' expr ']'                          # Indexacion
    | expr op=('*'|'/'|'%') expr                 # MulDivMod
    | '-' expr 				    # Negativo
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

    // ── MACHINE LEARNING ──────────────────────────────────────────
    | 'kNN' '(' expr ',' expr ',' expr ')'       # KNN
    | 'kMeans' '(' expr ',' expr ')'             # KMeans
    | 'normalizar' '(' expr ')'                  # Normalizar
    | 'errorCuadratico' '(' expr ',' expr ')'    # ErrorCuadratico
    | 'accuracy' '(' expr ',' expr ')'           # Accuracy

    // ── DEEP LEARNING ─────────────────────────────────────────────
    | 'crearRed' '(' expr ')'                    # CrearRed
    | 'entrenar' '(' expr ',' expr ',' expr ',' expr ')' # Entrenar
    | 'predecir' '(' expr ',' expr ')'           # Predecir
    | 'relu' '(' expr ')'                        # Relu
    | 'sigmoid' '(' expr ')'                     # Sigmoid
    | 'tanhAct' '(' expr ')'                     # TanhAct

    // ── IA PRE-ENTRENADA ──────────────────────────────────────────
    | 'sentimiento' '(' expr ')'                 # Sentimiento
    | 'predecirSerie' '(' expr ')'               # PredecirSerie
    | 'similitud' '(' expr ',' expr ')'          # Similitud
    | 'clasificarNumero' '(' expr ')'            # ClasificarNumero

    | 'stack' '(' ')'                            # StackCreation
    | lista                                      # ListaLiteral
    | diccionario                                # DiccionarioLiteral
    | matrix                                     # Matriz
    | functionCall                               # LlamadaFuncion
    | NUMBER                                     # Numero
    | BOOLEAN                                    # Booleano
    | STRING                                     # Cadena
    | ID                                         # Variable
    ;

// Listas y diccionarios
lista: '[' (expr (',' expr)*)? ']' ;
diccionario: '{' (expr ':' expr (',' expr ':' expr)*)? '}' ;

matrix: '[' fila (';' fila)* ']' ;
fila: expr (',' expr)* ;

functionCall: ID '(' (expr (',' expr)*)? ')' ;

FUNC: 'func';
NUMBER: [0-9]+ ('.' [0-9]+)? ;
BOOLEAN: 'true' | 'false' ;
STRING: '"' (~["\r\n])* '"' ;
ID: [a-zA-Z_][a-zA-Z0-9_]* ;
WS: [ \t\r\n]+ -> skip ;
COMMENT: '//' ~[\r\n]* -> skip ;

