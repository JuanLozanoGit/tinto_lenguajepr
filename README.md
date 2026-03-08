
# ☕ Tinto Language (tinto_lenguajepr)

![Python](https://img.shields.io/badge/Language-Python-blue?style=for-the-badge&logo=python)
![Status](https://img.shields.io/badge/Status-En%20Desarrollo-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Tinto** es un lenguaje de programación de juguete (*esolang*) con alma colombiana. Creado para transformar la lógica técnica en expresiones cercanas a nuestra cultura, permitiendo que programar sea tan natural como pedir un tinto en la esquina.

---

## 📖 Tabla de Contenidos
- [Características](#-características)
- [Arquitectura del Proyecto](#-arquitectura-del-proyecto)
- [Requisitos de Instalación](#-requisitos-de-instalación)
- [Guía de Sintaxis](#-guía-de-sintaxis)
- [Ejemplos de Código](#-ejemplos-de-código)
- [Créditos](#-créditos)

---

## ✨ Características
* **Identidad Local:** Palabras clave inspiradas en el español colombiano.
* **Basado en SLY:** Utiliza *Sly Lex-Yacc* para un análisis sintáctico robusto.
* **Ligero:** Ejecución directa sobre Python sin dependencias pesadas.
* **Educativo:** Ideal para entender cómo funciona un Lexer y un Parser de forma divertida.

---

## 🏗️ Arquitectura del Proyecto

El proyecto se divide en componentes modulares que gestionan el ciclo de vida del código:

* **`lexer.py`**: El "traductor" inicial. Convierte el texto en "tokens" (palabras que el lenguaje entiende).
* **`parser.py`**: El "arquitecto". Toma los tokens y verifica que sigan las reglas gramaticales de Tinto.
* **`main.py`**: El motor que pone todo en marcha.
* **`.tinto`**: Extensión de archivo oficial para tus programas.

---

## 🛠️ Requisitos de Instalación

Para "colar" este lenguaje en tu máquina, sigue estos pasos:

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/JuanLozanoGit/tinto_lenguajepr.git](https://github.com/JuanLozanoGit/tinto_lenguajepr.git)
    cd tinto_lenguajepr
    ```

2.  **Instalar dependencias:**
    Tinto requiere la librería `sly`.
    ```bash
    pip install sly
    ```

3.  **Ejecutar un programa:**
    ```bash
    python main.py ruta/de/tu/archivo.tinto
    ```

---

## 📝 Guía de Sintaxis

Tinto reemplaza la sintaxis rígida por una más fluida. Aquí tienes los comandos principales:

### 🔹 Definición de Variables
Usamos la palabra clave `sea` para asignar valores.
```tinto
sea granos = 50
sea marca = "Café de Colombia"

🔹 Mostrar en Pantalla
Para imprimir resultados, usamos el comando decir.
decir("El café está listo")
decir(granos)

🔹 Lógica Condicional
si (granos > 10) {
    decir("Hay suficiente para todos")
} sino {
    decir("Toca ir a la tienda")
}

🔹 Bucles (Iteraciones)
mientras (granos > 0) {
    decir("Moliendo grano...")
    granos = granos - 1
}

☕ Ejemplo Completo: "La Tienda de Doña Juana"
Guarda este código en un archivo llamado tienda.tinto:
sea presupuesto = 2000
sea precio_tinto = 500

decir("--- Bienvenido a la Cafetería ---")

si (presupuesto >= precio_tinto) {
    decir("Me regala un tinto, por favor.")
    presupuesto = presupuesto - precio_tinto
    decir("Saldo restante:")
    decir(presupuesto)
} sino {
    decir("No me alcanza ni para el aroma.")
}

🤝 Créditos
Este proyecto es una muestra de creatividad y pasión por las ciencias de la computación.
 * Autor: Juan Lozano
 * Inspiración: La cultura cafetera y el amor por los lenguajes de programación.
> "Programar es como hacer un buen tinto: requiere paciencia, técnica y un poquito de amor."
> 

¿Te gustaría que le añada alguna sección sobre cómo funcionan las expresiones regulares en el Lexer o prefieres dejarlo así de sencillo?

