import sys
import os
from antlr4 import *

# Forzamos la ruta de búsqueda
current_dir = os.path.dirname(os.path.abspath(__file__))
gen_dir = os.path.join(current_dir, 'antlr_gen')
sys.path.insert(0, gen_dir)

try:
    # Importamos los archivos generados directamente del path
    import TintoLexer
    import TintoParser
    from interpreter import TintoInterpreter
    print("✅ Motor TINTO cargado")
except ImportError as e:
    print(f"❌ Error: No se encontraron los archivos generados por ANTLR en {gen_dir}")
    print(f"Detalle: {e}")
    sys.exit(1)

def main(path_script):
    if not os.path.exists(path_script):
        print(f"Archivo no encontrado: {path_script}")
        return

    input_stream = FileStream(path_script, encoding='utf-8')
    lexer = TintoLexer.TintoLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = TintoParser.TintoParser(stream)
    tree = parser.program()
    
    visitor = TintoInterpreter()
    visitor.visit(tree)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python src/main.py scripts/prueba.tinto")
    else:
        main(sys.argv[1])