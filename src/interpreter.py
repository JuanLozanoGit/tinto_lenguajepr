import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), 'antlr_gen'))

from antlr4 import *


try:
    from antlr_gen.TintoVisitor import TintoVisitor
    from antlr_gen.TintoParser import TintoParser
    from antlr_gen.TintoLexer import TintoLexer
except ImportError:
    from TintoVisitor import TintoVisitor
    from TintoParser import TintoParser
    from TintoLexer import TintoLexer



class TintoInterpreter(TintoVisitor):
    def __init__(self):
        super().__init__()
        # El visitor ya tiene sus propias variables y funciones
        # Solo necesitamos inicializar si es necesario
        pass

    def main(self, archivo):
        try:
            input_stream = FileStream(archivo, encoding='utf-8')
        except Exception as e:
            print(f"Error al abrir archivo: {e}")
            return

        lexer = TintoLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = TintoParser(stream)

        try:
            tree = parser.program()
        except Exception as e:
            print(f"Error de sintaxis: {e}")
            return

        try:
            self.visit(tree)
        except Exception as e:
            print(f"Error durante la interpretación: {e}")
            import traceback
            traceback.print_exc()


def main():
    if len(sys.argv) < 2:
        print("Uso: python interpreter.py <archivo.tinto>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    interpreter = TintoInterpreter()
    interpreter.main(input_file)


if __name__ == '__main__':
    main()
