grammar Tinto;

program: statement+ EOF;

statement: assignment
         | ifStatement
         | whileStatement
         | forStatement
         | functionDecl
         | printStatement
         | graphStatement
         | expr ';' ;

assignment: ID '=' expr ';';

// Se definen bloques claros para el Visitor
ifStatement: 'si' '(' expr ')' '{' block '}' ('sino' '{' block '}')?;
whileStatement: 'mientras' '(' expr ')' '{' block '}';
forStatement: 'para' ID 'en' expr '..' expr '{' block '}';

block: statement+;

functionDecl: 'definir' ID '(' params? ')' '{' block 'retornar' expr ';' '}';

printStatement: 'mostrar' '(' expr ')' ';';
graphStatement: 'graficar' '(' expr ',' expr ')' ';';

params: ID (',' ID)*;

expr: '(' expr ')'                            # Parentesis
    | <assoc=right> expr '^' expr             # Potencia
    | expr op=('*'|'/'|'%') expr              # MultiplicacionDiv
    | expr op=('+'|'-') expr                  # SumaResta
    | expr op=('<'|'>'|'<='|'>='|'==') expr   # Comparacion
    | ID '(' argList? ')'                     # LlamadaFuncion
    | '[' row (';' row)* ']'                  # MatrizLiteral
    | ID                                      # Variable
    | NUMBER                                  # Numero
    | STRING                                  # Texto
    ;

row: expr (',' expr)*;
argList: expr (',' expr)*;

ID: [a-zA-Z_][a-zA-Z0-9_]*;
NUMBER: [0-9]+ ('.' [0-9]+)?;
STRING: '"' (~['"])* '"';
WS: [ \t\r\n]+ -> skip;
LINE_COMMENT: '#' ~[\r\n]* -> skip;
