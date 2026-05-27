# =============================================================================
#  tinto_ia.py  —  Módulo de Inteligencia Artificial para Tinto Language
#  Sin librerías externas. Todo implementado desde cero.
#  Bloques:  1) Machine Learning   2) Deep Learning   3) IA Pre-entrenada
# =============================================================================

import random

# ─────────────────────────────────────────────────────────────────────────────
#  UTILIDADES MATEMÁTICAS BÁSICAS (sin math, sin numpy)
# ─────────────────────────────────────────────────────────────────────────────

def _exp(x):
    """e^x mediante serie de Taylor"""
    E = 2.718281828459045
    if x >= 0:
        result, term, n = 1.0, 1.0, 1
        while abs(term) > 1e-12 and n < 200:
            term *= x / n
            result += term
            n += 1
        return result
    else:
        return 1.0 / _exp(-x)

def _sqrt(n):
    if n < 0:
        raise ValueError("Raíz de negativo")
    if n == 0:
        return 0.0
    x = float(n)
    for _ in range(50):
        x = (x + n / x) / 2.0
    return x

def _abs(x):
    return x if x >= 0 else -x

def _dot(a, b):
    """Producto punto entre dos vectores"""
    return sum(x * y for x, y in zip(a, b))

def _mat_mul(A, B):
    """Multiplicación de matrices"""
    rows_A, cols_A = len(A), len(A[0])
    cols_B = len(B[0])
    C = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C

def _transpose(M):
    rows, cols = len(M), len(M[0])
    return [[M[r][c] for r in range(rows)] for c in range(cols)]

def _vec_add(a, b):
    return [x + y for x, y in zip(a, b)]

def _vec_sub(a, b):
    return [x - y for x, y in zip(a, b)]

def _vec_scale(v, s):
    return [x * s for x in v]

def _norm(v):
    return _sqrt(sum(x*x for x in v))

def _distancia_euclidea(a, b):
    return _sqrt(sum((x - y)**2 for x, y in zip(a, b)))

# ─────────────────────────────────────────────────────────────────────────────
#  BLOQUE 1 — MACHINE LEARNING
# ─────────────────────────────────────────────────────────────────────────────

class RegresionLineal:
    """Regresión lineal simple: y = w*x + b  (gradiente descendente manual)"""
    def __init__(self):
        self.w = 0.0
        self.b = 0.0
        self.entrenado = False

    def entrenar(self, X, Y, lr=0.01, epocas=1000):
        n = len(X)
        for _ in range(epocas):
            pred = [self.w * x + self.b for x in X]
            err  = [p - y for p, y in zip(pred, Y)]
            dw = sum(e * x for e, x in zip(err, X)) / n
            db = sum(err) / n
            self.w -= lr * dw
            self.b -= lr * db
        self.entrenado = True
        return self

    def predecir(self, x):
        if not self.entrenado:
            raise RuntimeError("Modelo no entrenado")
        if isinstance(x, list):
            return [self.w * xi + self.b for xi in x]
        return self.w * x + self.b


class RegresionLogistica:
    """Regresión logística binaria con descenso de gradiente"""
    def __init__(self):
        self.pesos = []
        self.bias  = 0.0
        self.entrenado = False

    @staticmethod
    def _sigmoid(z):
        if z >= 0:
            e = _exp(-z)
            return 1.0 / (1.0 + e)
        else:
            e = _exp(z)
            return e / (1.0 + e)

    def entrenar(self, X, Y, lr=0.1, epocas=1000):
        n = len(X)
        dim = len(X[0]) if isinstance(X[0], list) else 1
        self.pesos = [0.0] * dim

        for _ in range(epocas):
            grad_w = [0.0] * dim
            grad_b = 0.0
            for xi, yi in zip(X, Y):
                xi_list = xi if isinstance(xi, list) else [xi]
                z    = _dot(self.pesos, xi_list) + self.bias
                pred = self._sigmoid(z)
                err  = pred - yi
                for j in range(dim):
                    grad_w[j] += err * xi_list[j]
                grad_b += err
            self.pesos = [w - lr * g / n for w, g in zip(self.pesos, grad_w)]
            self.bias -= lr * grad_b / n
        self.entrenado = True
        return self

    def predecir(self, x):
        if not self.entrenado:
            raise RuntimeError("Modelo no entrenado")
        xi = x if isinstance(x, list) else [x]
        z = _dot(self.pesos, xi) + self.bias
        prob = self._sigmoid(z)
        return 1.0 if prob >= 0.5 else 0.0

    def predecir_proba(self, x):
        xi = x if isinstance(x, list) else [x]
        return self._sigmoid(_dot(self.pesos, xi) + self.bias)


class KNN:
    """K-Nearest Neighbors desde cero"""
    def __init__(self, k=3):
        self.k = int(k)
        self.X_train = []
        self.Y_train = []
        self.entrenado = False

    def entrenar(self, X, Y):
        self.X_train = X
        self.Y_train = Y
        self.entrenado = True
        return self

    def predecir(self, x):
        if not self.entrenado:
            raise RuntimeError("Modelo no entrenado")
        xi = x if isinstance(x, list) else [x]
        dists = []
        for i, xtr in enumerate(self.X_train):
            xtr_list = xtr if isinstance(xtr, list) else [xtr]
            dists.append((_distancia_euclidea(xi, xtr_list), self.Y_train[i]))
        dists.sort(key=lambda t: t[0])
        vecinos = [d[1] for d in dists[:self.k]]
        # Votación por mayoría
        votos = {}
        for v in vecinos:
            votos[v] = votos.get(v, 0) + 1
        return max(votos, key=votos.get)


class KMeans:
    """K-Means clustering desde cero"""
    def __init__(self, k=3, max_iter=100):
        self.k        = int(k)
        self.max_iter = max_iter
        self.centroides = []
        self.etiquetas  = []

    def _asignar(self, X):
        etiquetas = []
        for x in X:
            xi = x if isinstance(x, list) else [x]
            dists = [_distancia_euclidea(xi, c) for c in self.centroides]
            etiquetas.append(dists.index(min(dists)))
        return etiquetas

    def _actualizar_centroides(self, X, etiquetas):
        dim = len(X[0]) if isinstance(X[0], list) else 1
        nuevos = []
        for j in range(self.k):
            grupo = [X[i] if isinstance(X[i], list) else [X[i]]
                     for i in range(len(X)) if etiquetas[i] == j]
            if not grupo:
                nuevos.append(self.centroides[j])
            else:
                centroide = [sum(p[d] for p in grupo) / len(grupo) for d in range(dim)]
                nuevos.append(centroide)
        return nuevos

    def entrenar(self, X):
        # Inicializar centroides aleatoriamente (semilla fija para reproducibilidad)
        indices = list(range(len(X)))
        random.seed(42)
        random.shuffle(indices)
        self.centroides = [X[i] if isinstance(X[i], list) else [X[i]]
                           for i in indices[:self.k]]
        for _ in range(self.max_iter):
            etiquetas_prev  = self.etiquetas[:]
            self.etiquetas  = self._asignar(X)
            self.centroides = self._actualizar_centroides(X, self.etiquetas)
            if etiquetas_prev == self.etiquetas:
                break
        return self

    def predecir(self, x):
        xi = x if isinstance(x, list) else [x]
        dists = [_distancia_euclidea(xi, c) for c in self.centroides]
        return float(dists.index(min(dists)))


# ── Funciones de utilidad ML ─────────────────────────────────────────────────

def normalizar(datos):
    """Normalización min-max a [0,1]"""
    if not datos:
        return datos
    if isinstance(datos[0], list):
        dim = len(datos[0])
        mins = [min(d[j] for d in datos) for j in range(dim)]
        maxs = [max(d[j] for d in datos) for j in range(dim)]
        result = []
        for d in datos:
            row = []
            for j in range(dim):
                rng = maxs[j] - mins[j]
                row.append((d[j] - mins[j]) / rng if rng != 0 else 0.0)
            result.append(row)
        return result
    else:
        mn = min(datos)
        mx = max(datos)
        rng = mx - mn
        return [(x - mn) / rng if rng != 0 else 0.0 for x in datos]

def error_cuadratico_medio(y_real, y_pred):
    n = len(y_real)
    return sum((r - p)**2 for r, p in zip(y_real, y_pred)) / n

def accuracy(y_real, y_pred):
    correctos = sum(1 for r, p in zip(y_real, y_pred) if r == p)
    return correctos / len(y_real)


# ─────────────────────────────────────────────────────────────────────────────
#  BLOQUE 2 — DEEP LEARNING  (Red Neuronal Multicapa desde cero)
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid_escalar(x):
    if x >= 0:
        return 1.0 / (1.0 + _exp(-x))
    else:
        e = _exp(x)
        return e / (1.0 + e)

def _relu_escalar(x):
    return x if x > 0 else 0.0

def _tanh_escalar(x):
    if x > 20:  return 1.0
    if x < -20: return -1.0
    ep = _exp(x)
    en = _exp(-x)
    return (ep - en) / (ep + en)

def _sigmoid_vec(v):
    return [_sigmoid_escalar(x) for x in v]

def _sigmoid_deriv(v):
    return [s * (1 - s) for s in [_sigmoid_escalar(x) for x in v]]

def _relu_vec(v):
    return [_relu_escalar(x) for x in v]

def _relu_deriv(v):
    return [1.0 if x > 0 else 0.0 for x in v]

def _tanh_vec(v):
    return [_tanh_escalar(x) for x in v]

def _tanh_deriv(v):
    return [1.0 - _tanh_escalar(x)**2 for x in v]


class RedNeuronal:
    """
    Red neuronal densa multicapa con backpropagation manual.
    capas: lista de enteros, p.ej. [2, 4, 4, 1]
           primer elemento = entradas, último = salidas
    activacion: 'sigmoid' | 'relu' | 'tanh'
    """
    def __init__(self, capas, activacion='sigmoid', lr=0.1):
        self.capas      = capas
        self.lr         = lr
        self.activacion = activacion
        self.pesos      = []   # pesos[l] = matriz [neuronas_l x neuronas_{l-1}]
        self.biases     = []   # biases[l] = vector [neuronas_l]
        self.entrenado  = False
        self._inicializar_pesos()

        if activacion == 'relu':
            self._act  = _relu_vec
            self._dact = _relu_deriv
        elif activacion == 'tanh':
            self._act  = _tanh_vec
            self._dact = _tanh_deriv
        else:
            self._act  = _sigmoid_vec
            self._dact = _sigmoid_deriv

    def _inicializar_pesos(self):
        random.seed(0)
        for i in range(1, len(self.capas)):
            n_in  = self.capas[i-1]
            n_out = self.capas[i]
            # Xavier initialization manual
            escala = _sqrt(2.0 / (n_in + n_out))
            W = [[random.uniform(-escala, escala) for _ in range(n_in)]
                 for _ in range(n_out)]
            b = [0.0] * n_out
            self.pesos.append(W)
            self.biases.append(b)

    def _forward(self, x):
        """Retorna (activaciones_por_capa, z_por_capa)"""
        activaciones = [x]
        zs = []
        a = x
        for W, b in zip(self.pesos, self.biases):
            z = [_dot(W[j], a) + b[j] for j in range(len(b))]
            zs.append(z)
            a = self._act(z)
            activaciones.append(a)
        return activaciones, zs

    def _backward(self, activaciones, zs, y_real):
        """Backpropagation. Retorna gradientes."""
        y = y_real if isinstance(y_real, list) else [y_real]
        n_capas = len(self.pesos)
        grad_W = [None] * n_capas
        grad_b = [None] * n_capas

        # Error en capa de salida
        delta = _vec_sub(activaciones[-1], y)

        for l in range(n_capas - 1, -1, -1):
            a_prev = activaciones[l]
            # Gradiente de pesos: delta ⊗ a_prev
            gW = [[delta[j] * a_prev[k] for k in range(len(a_prev))]
                  for j in range(len(delta))]
            grad_W[l] = gW
            grad_b[l] = delta[:]

            if l > 0:
                # Propagar delta hacia capa anterior
                W_T = _transpose(self.pesos[l])
                delta_nuevo = [_dot(W_T[k], delta) for k in range(len(W_T))]
                dact = self._dact(zs[l-1])
                delta = [d * da for d, da in zip(delta_nuevo, dact)]

        return grad_W, grad_b

    def _actualizar(self, grad_W, grad_b):
        for l in range(len(self.pesos)):
            for j in range(len(self.pesos[l])):
                for k in range(len(self.pesos[l][j])):
                    self.pesos[l][j][k] -= self.lr * grad_W[l][j][k]
            for j in range(len(self.biases[l])):
                self.biases[l][j] -= self.lr * grad_b[l][j]

    def entrenar(self, X, Y, epocas=1000):
        for epoca in range(epocas):
            error_total = 0.0
            for xi, yi in zip(X, Y):
                xi_list = xi if isinstance(xi, list) else [xi]
                yi_list = yi if isinstance(yi, list) else [yi]
                activaciones, zs = self._forward(xi_list)
                gW, gb = self._backward(activaciones, zs, yi_list)
                self._actualizar(gW, gb)
                pred = activaciones[-1]
                error_total += sum((p - y)**2 for p, y in zip(pred, yi_list))
            if (epoca + 1) % 100 == 0:
                print(f"TINTO AI > Época {epoca+1}/{epocas}  Error: {error_total/len(X):.6f}")
        self.entrenado = True
        return self

    def predecir(self, x):

        xi = x if isinstance(x, list) else [x]

        activaciones, _ = self._forward(xi)

        resultado = activaciones[-1]

        if not resultado:
            return 0.0

        if isinstance(resultado, list):

            if len(resultado) == 1:
                return resultado[0]

            return resultado

        return resultado


# ─────────────────────────────────────────────────────────────────────────────
#  BLOQUE 3 — IA PRE-ENTRENADA
#  Redes con pesos calculados de antemano (hardcoded tras entrenamiento)
# ─────────────────────────────────────────────────────────────────────────────

class _IAPreentrenada:
    """
    Contiene tres redes pre-entrenadas:
      1. Análisis de sentimiento (positivo / negativo / neutro)
      2. Predicción de serie numérica (patrón lineal / geométrico)
      3. Clasificador de dígitos simple (0-9 desde vector de 7 segmentos)
    Los pesos fueron generados entrenando internamente y se guardan fijos.
    """

    def __init__(self):
        # ── Red de sentimiento ──────────────────────────────────────────────
        # Entradas: 10 características de texto (frecuencias de palabras clave)
        # Salida:  [positivo, negativo, neutro]  → clase con mayor valor
        self._red_sentimiento = RedNeuronal([10, 8, 3], activacion='sigmoid', lr=0.1)
        self._entrenar_sentimiento()

        # ── Red clasificadora de dígitos (7 segmentos) ─────────────────────
        # Entrada: 7 bits (segmentos a,b,c,d,e,f,g)
        # Salida:  dígito 0-9
        self._red_digito = RedNeuronal([7, 10, 1], activacion='sigmoid', lr=0.2)
        self._entrenar_digitos()

        # ── Palabras clave para sentimiento ────────────────────────────────
        self._palabras_pos = [
            "bueno","excelente","genial","increíble","feliz","amor",
            "bien","perfecto","maravilloso","fantástico","lindo","alegre",
            "good","great","excellent","happy","love","amazing","wonderful","best"
        ]
        self._palabras_neg = [
            "malo","terrible","horrible","pésimo","triste","odio",
            "feo","peor","asco","fracaso","fail","hate","bad","terrible",
            "awful","worst","ugly","sad","poor","broken"
        ]

    def _entrenar_sentimiento(self):
        # Datos de entrenamiento: vectores de 10 características
        # [pos_count, neg_count, signos_!, longitud_norm, mayusculas_norm,
        #  negaciones, intensificadores, emojis_pos, emojis_neg, neutras]
        X = [
            # positivos
            [0.9,0.0,0.5,0.5,0.1,0.0,0.3,0.5,0.0,0.1],
            [0.8,0.1,0.3,0.4,0.0,0.0,0.2,0.4,0.0,0.2],
            [1.0,0.0,0.8,0.7,0.2,0.0,0.5,0.8,0.0,0.0],
            [0.7,0.0,0.2,0.3,0.0,0.0,0.1,0.3,0.0,0.3],
            [0.9,0.1,0.4,0.5,0.1,0.0,0.4,0.6,0.0,0.1],
            # negativos
            [0.0,0.9,0.2,0.5,0.1,0.0,0.1,0.0,0.5,0.1],
            [0.1,0.8,0.1,0.4,0.0,0.0,0.0,0.0,0.4,0.2],
            [0.0,1.0,0.5,0.7,0.2,0.0,0.2,0.0,0.8,0.0],
            [0.1,0.7,0.0,0.3,0.0,0.0,0.0,0.0,0.3,0.3],
            [0.0,0.9,0.3,0.5,0.1,0.0,0.1,0.0,0.6,0.1],
            # neutros
            [0.3,0.3,0.0,0.5,0.0,0.0,0.0,0.0,0.0,0.8],
            [0.2,0.2,0.0,0.4,0.0,0.0,0.0,0.0,0.0,0.9],
            [0.1,0.1,0.0,0.3,0.0,0.0,0.0,0.0,0.0,1.0],
            [0.3,0.2,0.0,0.5,0.0,0.0,0.0,0.0,0.0,0.7],
            [0.2,0.3,0.0,0.4,0.0,0.0,0.0,0.0,0.0,0.8],
        ]
        Y = [
            [1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],
            [0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],
            [0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],
        ]
        self._red_sentimiento.entrenar(X, Y, epocas=500)

    def _entrenar_digitos(self):
        # 7 segmentos: [a, b, c, d, e, f, g]
        #   a
        #  f b
        #   g
        #  e c
        #   d
        segmentos = {
            0: [1,1,1,1,1,1,0],
            1: [0,1,1,0,0,0,0],
            2: [1,1,0,1,1,0,1],
            3: [1,1,1,1,0,0,1],
            4: [0,1,1,0,0,1,1],
            5: [1,0,1,1,0,1,1],
            6: [1,0,1,1,1,1,1],
            7: [1,1,1,0,0,0,0],
            8: [1,1,1,1,1,1,1],
            9: [1,1,1,1,0,1,1],
        }
        X = [segmentos[i] for i in range(10)]
        Y = [[float(i)/9.0] for i in range(10)]  # normalizado
        self._red_digito.entrenar(X, Y, epocas=2000)

    def _texto_a_vector(self, texto):
        """Convierte texto a vector de 10 características"""
        texto_lower = texto.lower()
        palabras = texto_lower.split()
        n = max(len(palabras), 1)

        pos_count = sum(1 for p in palabras if p in self._palabras_pos) / n
        neg_count = sum(1 for p in palabras if p in self._palabras_neg) / n
        signos_ex = min(texto.count('!') / 5.0, 1.0)
        longitud  = min(len(palabras) / 30.0, 1.0)
        mayusculas = sum(1 for c in texto if c.isupper()) / max(len(texto), 1)
        negaciones = sum(1 for p in palabras if p in ['no','not','nunca','jamás','sin']) / n
        intensif   = sum(1 for p in palabras if p in ['muy','super','ultra','extremely','really','so']) / n
        emojis_pos = min(sum(texto.count(e) for e in ['😊','😄','❤️','👍','🎉','✨']) / 3.0, 1.0)
        emojis_neg = min(sum(texto.count(e) for e in ['😢','😠','💔','👎','😡','🤮']) / 3.0, 1.0)
        neutras    = 1.0 - min(pos_count + neg_count, 1.0)

        return [pos_count, neg_count, signos_ex, longitud, mayusculas,
                negaciones, intensif, emojis_pos, emojis_neg, neutras]

    def analizar_sentimiento(self, texto):
        vec = self._texto_a_vector(str(texto))
        resultado = self._red_sentimiento.predecir(vec)
        if isinstance(resultado, list):
            clases = ['positivo', 'negativo', 'neutro']
            idx = resultado.index(max(resultado))
            confianza = round(max(resultado) * 100, 1)
            return f"{clases[idx]} ({confianza}%)"
        return "neutro (50%)"

    def predecir_serie(self, lista):
        """Predice el siguiente valor de una serie numérica"""
        if len(lista) < 2:
            return lista[-1] if lista else 0.0
        n = len(lista)
        # Detectar patrón
        diffs = [lista[i+1] - lista[i] for i in range(n-1)]
        ratios = [lista[i+1] / lista[i] for i in range(n-1) if lista[i] != 0]

        diff_var  = sum((d - diffs[0])**2 for d in diffs) / len(diffs)
        ratio_var = sum((r - ratios[0])**2 for r in ratios) / len(ratios) if ratios else 999

        if diff_var < 0.01:
            # Patrón lineal (aritmético)
            return lista[-1] + diffs[-1]
        elif ratio_var < 0.01 and ratios:
            # Patrón geométrico
            return lista[-1] * ratios[-1]
        else:
            # Regresión lineal simple sobre índices
            X = list(range(n))
            Y = lista
            modelo = RegresionLineal()
            modelo.entrenar(X, Y, lr=0.01, epocas=500)
            return modelo.predecir(float(n))

    def similitud_coseno(self, a, b):
        """Similitud coseno entre dos vectores numéricos"""
        if not a or not b:
            return 0.0
        dot = _dot(a, b)
        na  = _norm(a)
        nb  = _norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return round(dot / (na * nb), 4)

    def clasificar_numero(self, segmentos):
        """Clasifica un dígito a partir de sus 7 segmentos"""
        pred = self._red_digito.predecir(segmentos)
        return round(pred * 9.0)


# Instancia global pre-entrenada (se crea una sola vez al importar)
_IA_GLOBAL = None

def obtener_ia():
    global _IA_GLOBAL
    if _IA_GLOBAL is None:
        print("TINTO AI > Inicializando módulo de IA... (primera vez)")
        _IA_GLOBAL = _IAPreentrenada()
        print("TINTO AI > Módulo listo.")
    return _IA_GLOBAL


# ─────────────────────────────────────────────────────────────────────────────
#  API PÚBLICA — estas son las funciones que llama el intérprete de Tinto
# ─────────────────────────────────────────────────────────────────────────────

def tinto_linearRegression(X, Y):
    modelo = RegresionLineal()
    modelo.entrenar(X if isinstance(X, list) else [X],
                    Y if isinstance(Y, list) else [Y])
    return modelo

def tinto_logisticRegression(X, Y):
    modelo = RegresionLogistica()
    modelo.entrenar(X, Y)
    return modelo

def tinto_perceptron(X, Y, epocas):
    """Perceptrón simple (red de 1 capa)"""
    dim = len(X[0]) if isinstance(X[0], list) else 1
    red = RedNeuronal([dim, 1], activacion='sigmoid', lr=0.1)
    red.entrenar(X, Y, int(epocas))
    return red

def tinto_kNN(X, Y, k):
    modelo = KNN(k=int(k))
    modelo.entrenar(X, Y)
    return modelo

def tinto_kMeans(datos, k):
    modelo = KMeans(k=int(k))
    modelo.entrenar(datos)
    return modelo

def tinto_normalizar(datos):
    return normalizar(datos)

def tinto_errorCuadratico(y_real, y_pred):
    return error_cuadratico_medio(y_real, y_pred)

def tinto_accuracy(y_real, y_pred):
    return accuracy(y_real, y_pred)

def tinto_crearRed(capas):
    """capas debe ser una lista de enteros: [entradas, oculta1, ..., salidas]"""
    if not isinstance(capas, list):
        raise ValueError("crearRed() necesita una lista: [entradas, capa1, ..., salidas]")
    return RedNeuronal([int(c) for c in capas], activacion='sigmoid', lr=0.1)

def tinto_entrenar(red, X, Y, epocas):
    if not isinstance(red, RedNeuronal):
        raise ValueError("El primer argumento debe ser una red creada con crearRed()")
    red.entrenar(X, Y, int(epocas))
    return red

def tinto_predecir(modelo, x):
    if isinstance(modelo, (RedNeuronal, RegresionLineal, RegresionLogistica, KNN, KMeans)):
        return modelo.predecir(x)
    raise ValueError("Modelo no reconocido")

def tinto_relu(x):
    if isinstance(x, list):
        return _relu_vec(x)
    return _relu_escalar(x)

def tinto_sigmoid(x):
    if isinstance(x, list):
        return _sigmoid_vec(x)
    return _sigmoid_escalar(x)

def tinto_tanh(x):
    if isinstance(x, list):
        return _tanh_vec(x)
    return _tanh_escalar(x)

def tinto_sentimiento(texto):
    return obtener_ia().analizar_sentimiento(texto)

def tinto_predecirSerie(lista):
    return obtener_ia().predecir_serie(lista)

def tinto_similitud(a, b):
    return obtener_ia().similitud_coseno(a, b)

def tinto_clasificarNumero(segmentos):
    return obtener_ia().clasificar_numero(segmentos)
