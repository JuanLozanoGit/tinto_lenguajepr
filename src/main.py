import sys
import os
from antlr4 import *

# Configuración de rutas para encontrar los archivos generados
current_dir = os.path.dirname(os.path.abspath(__file__))
# Si tu estructura es src/main.py y src/antlr_gen/...
gen_dir = os.path.join(current_dir, 'antlr_gen')
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
    # Validar existencia del archivo
    if not os.path.exists(path_script):
        print(f"Error: El script '{path_script}' no existe.")
        return

    try:
        # 1. Flujo de lectura
        input_stream = FileStream(path_script, encoding='utf-8') [cite: 1]
        
        # 2. Análisis Léxico
        lexer = TintoLexer.TintoLexer(input_stream)
        stream = CommonTokenStream(lexer)
        
        # 3. Análisis Sintáctico
        parser = TintoParser.TintoParser(stream) [cite: 1]
        tree = parser.program() [cite: 1]
        
        # Verificar si hubo errores de sintaxis antes de ejecutar
        if parser.getNumberOfSyntaxErrors() > 0:
            print(f"❌ Se encontraron {parser.getNumberOfSyntaxErrors()} errores sintácticos.")
            return

        # 4. Ejecución con el Visitor
        visitor = TintoInterpreter()
        visitor.visit(tree) [cite: 1]
        
    except Exception as e:
        print(f"Error durante la ejecución: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Ejemplo de uso si no se pasan argumentos
        print("Uso sugerido: python main.py scripts/prueba.tinto")
    else:
        main(sys.argv[1])
