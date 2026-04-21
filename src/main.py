import sys
import os
from antlr4 import *

current_dir = os.path.dirname(os.path.abspath(__file__))
# Ruta a la subcarpeta que contiene los generados
gen_dir = os.path.join(current_dir, 'antlr_gen', 'grammar')
sys.path.insert(0, gen_dir)

try:
    import TintoLexer
    import TintoParser
    from interpreter import TintoInterpreter
    print("Motor TINTO v1.0 - Listo")
except ImportError as e:
    print(f"Error crítico: No se encontraron los archivos de ANTLR en {gen_dir}")
    print(f"Detalle: {e}")
    sys.exit(1)

def main(path_script):
    if not os.path.exists(path_script):
        print(f"Error: El script '{path_script}' no existe.")
        return
    try:
        input_stream = FileStream(path_script, encoding='utf-8')
        lexer = TintoLexer.TintoLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = TintoParser.TintoParser(stream)
        tree = parser.program()
        if parser.getNumberOfSyntaxErrors() > 0:
            print(f"Se encontraron {parser.getNumberOfSyntaxErrors()} errores sintácticos.")
            return
        visitor = TintoInterpreter()
        visitor.visit(tree)
    except Exception as e:
        print(f"Error durante la ejecución: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso sugerido: python main.py scripts/prueba.tinto")
    else:
        main(sys.argv[1])
