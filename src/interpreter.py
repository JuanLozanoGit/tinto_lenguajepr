class TintoInterpreter(TintoVisitor):
    def __init__(self):
        super().__init__()
        # Tabla de símbolos para guardar valores
        self.variables = {}

    # --- Lógica de Control de Flujo ---

    def visitIf_stat(self, ctx):
        # Evaluar la condición
        condition = self.visit(ctx.expr())
        
        if condition:
            self.visit(ctx.block(0))
        elif ctx.else_block: # Si existe una segunda cláusula block (else)
            self.visit(ctx.block(1))

    def visitWhile_stat(self, ctx):
        # Ejecutar mientras la condición sea verdadera
        while self.visit(ctx.expr()):
            self.visit(ctx.block())

    # --- Lógica de Evaluación de Expresiones ---

    def visitComparison(self, ctx):
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        op = ctx.getChild(1).getText()
        
        if op == '==': return left == right
        if op == '!=': return left != right
        if op == '>':  return left > right
        if op == '<':  return left < right
        return False

    def visitNumber(self, ctx):
        return int(ctx.getText())

    def visitID(self, ctx):
        return self.variables.get(ctx.getText(), 0)

    # --- Lógica de Bloques ---
    
    def visitBlock(self, ctx):
        for stmt in ctx.statement():
            self.visit(stmt)
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
