# Tinto Language

## Descripción general

**Tinto Language** es un lenguaje de programación interpretado diseñado con fines educativos para demostrar cómo construir un lenguaje desde cero utilizando **Python** y **ANTLR** (Another Tool for Language Recognition). El proyecto implementa un **analizador léxico**, un **parser** y un **intérprete** que permiten ejecutar programas escritos en el lenguaje **Tinto**.

El objetivo del proyecto es mostrar el proceso completo de construcción de un lenguaje: definición de la gramática, generación del parser, y ejecución de instrucciones mediante un intérprete.

---

# Arquitectura del proyecto

El sistema se compone de tres componentes principales:

**1. Gramática del lenguaje**

El archivo:

```
grammar/Tinto.g4
```

define las reglas sintácticas del lenguaje utilizando **ANTLR4**.
A partir de esta gramática se generan automáticamente los archivos del parser.

La gramática describe:

* estructura del programa
* reglas de expresiones
* operadores
* asignaciones
* evaluaciones matemáticas

---

**2. Parser y Lexer**

ANTLR utiliza la gramática para generar:

* **Lexer** → divide el código en tokens
* **Parser** → valida la estructura del programa

Estos archivos se encuentran dentro de:

```
src/antlr_gen/
```

---

**3. Intérprete**

El archivo:

```
src/interpreter.py
```

implementa la lógica que ejecuta las instrucciones del lenguaje.

Este componente:

* evalúa expresiones
* maneja variables
* ejecuta operaciones matemáticas
* interpreta la estructura generada por el parser

---

# Estructura del proyecto

```
tinto_lenguajepr-main
│
├── grammar
│   └── Tinto.g4
│
├── scripts
│   └── prueba.tinto
│
├── src
│   ├── main.py
│   ├── interpreter.py
│   └── antlr_gen
│
├── requirements.txt
├── README.md
└── antlr-4.13.1-complete.jar
```

Descripción de los componentes principales:

* **grammar/** → definición de la sintaxis del lenguaje
* **scripts/** → programas de ejemplo escritos en Tinto
* **src/main.py** → punto de entrada del intérprete
* **src/interpreter.py** → implementación del intérprete
* **antlr_gen/** → archivos generados automáticamente por ANTLR
* **requirements.txt** → dependencias del proyecto

---

# Requisitos

Antes de ejecutar el proyecto se requiere instalar:

* **Python 3.8 o superior**
* **Java JDK** (para ANTLR)
* **ANTLR 4**
* Sistema operativo Linux o macOS

---

# Instalación del proyecto

Primero se debe clonar o descargar el repositorio.

```
git clone <url-del-repositorio>
cd tinto_lenguajepr-main
```

Si el proyecto fue descargado como `.zip`, simplemente se debe descomprimir y entrar a la carpeta.

---

# Crear y activar el entorno virtual

## En Ubuntu

Crear entorno virtual:

```
python3 -m venv venv
```

Activar entorno virtual:

```
source venv/bin/activate
```

Instalar dependencias:

```
pip install -r requirements.txt
```

---

## En macOS (Mac M1 / M2 / M3)

Crear entorno virtual:

```
python3 -m venv venv
```

Activar entorno virtual:

```
source venv/bin/activate
```

Instalar dependencias:

```
pip3 install -r requirements.txt
```

---

# Ejecutar el lenguaje

El programa principal del lenguaje es:

```
src/main.py
```

Para ejecutar un programa escrito en **Tinto**, se usa el siguiente comando:

```
python3 src/main.py scripts/prueba.tinto
```

El intérprete leerá el archivo `.tinto`, procesará la gramática definida en ANTLR y ejecutará las instrucciones.

---

# Ejemplo de programa en Tinto

Archivo:

```
scripts/prueba.tinto
```

Ejemplo de código:

```
a = 5
b = 3
a + b
```

Resultado esperado:

```
8
```

El intérprete evalúa las expresiones matemáticas y muestra el resultado en consola.

---

# Cómo funciona el lenguaje

El flujo de ejecución del lenguaje sigue las siguientes etapas:

**1. Lectura del programa**

El archivo `.tinto` es cargado por `main.py`.

**2. Análisis léxico**

ANTLR convierte el código en tokens que representan:

* números
* operadores
* identificadores
* símbolos del lenguaje

**3. Análisis sintáctico**

El parser verifica que el programa siga las reglas definidas en `Tinto.g4`.

**4. Interpretación**

El intérprete recorre el árbol sintáctico generado por el parser y ejecuta las instrucciones del programa.

---

# Características del lenguaje

El lenguaje **Tinto** incluye las siguientes características:

* interpretación directa de código
* soporte para **variables**
* operaciones matemáticas
* evaluación de expresiones
* sintaxis simple orientada a aprendizaje
* implementación basada en **ANTLR4**

