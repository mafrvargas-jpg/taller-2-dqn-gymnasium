# Taller 2 — Resolver un ambiente nuevo de Gymnasium con DQN

##Grupo 4 

Implementación de un agente **Deep Q-Network (DQN)** en el ambiente `LunarLander-v3` de Gymnasium.

El objetivo del proyecto es entrenar un agente de aprendizaje por refuerzo que aprenda a controlar una nave espacial para realizar aterrizajes exitosos, buscando maximizar la recompensa acumulada.

---

# 1. Ambiente seleccionado: LunarLander-v3

Para este proyecto se seleccionó el ambiente `LunarLander-v3` de Gymnasium. El objetivo del ambiente es controlar una nave espacial para lograr un aterrizaje seguro en la plataforma, utilizando un conjunto discreto de acciones.

El ambiente fue seleccionado porque presenta un problema de control con un espacio de observaciones continuo y un espacio de acciones discreto, características que lo hacen adecuado para implementar un agente basado en Deep Q-Network (DQN).

## 1.1 Espacio de observaciones

El ambiente proporciona un vector de observaciones de **8 variables**, con tipo de dato `float32`:

| Índice | Variable | Rango |
|---|---|---|
| 0 | Posición horizontal (X) | [-2.5, 2.5] |
| 1 | Posición vertical (Y) | [-2.5, 2.5] |
| 2 | Velocidad horizontal (X) | [-10, 10] |
| 3 | Velocidad vertical (Y) | [-10, 10] |
| 4 | Ángulo de la nave | [-2π, 2π] |
| 5 | Velocidad angular | [-10, 10] |
| 6 | Contacto de la pierna izquierda | [0, 1] |
| 7 | Contacto de la pierna derecha | [0, 1] |

El espacio de observaciones corresponde a un `Box` de dimensión `(8,)` y tipo `float32`.

Por lo tanto, la entrada de la red neuronal utilizada por el DQN tendrá una dimensión de **8 características**.

## 1.2 Espacio de acciones

El espacio de acciones es discreto:

~~~
Discrete(4)
~~~

El agente puede seleccionar una de las siguientes cuatro acciones:

| Acción | Descripción |
|---|---|
| 0 | No realizar ninguna acción |
| 1 | Activar el motor de orientación izquierdo |
| 2 | Activar el motor principal |
| 3 | Activar el motor de orientación derecho |

Estas acciones permiten al agente controlar la orientación y el movimiento de la nave durante el proceso de aterrizaje.

## 1.3 Sistema de recompensas

LunarLander-v3 utiliza un sistema de recompensas diseñado para orientar progresivamente al agente hacia un aterrizaje exitoso.

La recompensa considera diferentes aspectos del comportamiento de la nave, entre ellos:

- Distancia de la nave respecto a la plataforma de aterrizaje.
- Velocidad de la nave.
- Inclinación de la nave.
- Contacto de las piernas con la plataforma.
- Consumo de combustible asociado al uso de los motores.

Adicionalmente:

- Cada contacto de una pierna con la plataforma genera una recompensa positiva.
- El uso de los motores laterales genera un costo por frame.
- El uso del motor principal genera un costo por frame.
- Un aterrizaje exitoso genera una recompensa terminal de **+100**.
- Un choque genera una recompensa terminal de **-100**.

Por lo tanto, el objetivo del agente no consiste únicamente en llegar a la plataforma, sino en aprender a controlar la nave de manera estable y eficiente para maximizar la recompensa acumulada.

## 1.4 Condiciones de terminación

La API de Gymnasium devuelve información sobre la finalización de cada episodio mediante las variables:

~~~
observation, reward, terminated, truncated, info = env.step(action)
~~~

Estas variables se interpretan de la siguiente manera:

- `terminated`: indica que el episodio terminó debido a una condición propia del ambiente, como un aterrizaje exitoso o un choque.
- `truncated`: indica que el episodio terminó debido a una condición externa, como alcanzar el límite máximo de pasos.

Durante la exploración del ambiente se verificó el comportamiento de ambas variables.

En una de las pruebas realizadas, el episodio terminó después de **119 pasos**, con:

~~~
Terminated: True
Truncated: False
~~~

Esto permitió comprobar el funcionamiento de la API de finalización de episodios de Gymnasium.

## 1.5 Exploración inicial del ambiente

Antes de implementar el agente DQN se realizó una exploración inicial de `LunarLander-v3` para identificar sus principales características y verificar la información recibida por el agente.

Durante esta etapa se comprobó:

- Espacio de observaciones.
- Dimensión de las observaciones.
- Tipo de dato de las observaciones.
- Espacio de acciones.
- Funcionamiento de las cuatro acciones disponibles.
- Recompensas obtenidas al ejecutar diferentes acciones.
- Condiciones de terminación del episodio.

La observación utilizada por el agente tiene la siguiente estructura:

~~~
[posición_x, posición_y, velocidad_x, velocidad_y,
 ángulo, velocidad_angular, contacto_pierna_izquierda,
 contacto_pierna_derecha]
~~~

Las pruebas realizadas mostraron que diferentes acciones pueden producir recompensas diferentes dependiendo del estado de la nave.

Esto evidencia que el agente debe aprender una política que relacione el estado observado con la acción que permita maximizar la recompensa acumulada.

Los resultados y códigos correspondientes a esta exploración se encuentran en el notebook `Taller_2_DQN_LunarLander.ipynb`.

## 1.6 Preprocesamiento de las observaciones

Las variables que conforman el estado presentan diferentes escalas.

Por ejemplo:

- Las posiciones tienen un rango aproximado de `[-2.5, 2.5]`.
- Las velocidades pueden alcanzar valores entre `[-10, 10]`.
- El ángulo presenta un rango de `[-2π, 2π]`.
- Las variables de contacto toman valores entre `0` y `1`.

Debido a estas diferencias de escala, se considera conveniente realizar un proceso de normalización de las observaciones antes de suministrarlas a la red neuronal.

La normalización permite que las diferentes características tengan magnitudes comparables y puede facilitar el proceso de optimización de la red neuronal.

La transformación específica utilizada será definida de acuerdo con la implementación final del agente DQN.

## 1.7 Conclusiones de la exploración del ambiente

La exploración realizada permitió identificar las principales características del ambiente `LunarLander-v3` y comprender la información que estará disponible para el agente durante el proceso de aprendizaje.

El ambiente cuenta con un espacio de observaciones continuo compuesto por 8 variables de tipo `float32`, relacionadas con la posición, velocidad, orientación y contacto de las piernas de la nave. Asimismo, cuenta con un espacio de acciones discreto compuesto por 4 acciones posibles.

Las pruebas realizadas permitieron verificar el funcionamiento del ambiente, las características de las observaciones, las acciones disponibles y el comportamiento de las recompensas ante diferentes acciones.

A partir de esta exploración se determinó que `LunarLander-v3` es adecuado para la implementación de un agente Deep Q-Network (DQN), debido a que presenta un estado representado mediante un vector de características y un conjunto discreto de acciones.

---

# 2. Implementación del agente DQN

> **Esta sección corresponde a la Persona 2.**

## 2.1 Descripción general

En esta sección se documentará la implementación del agente **Deep Q-Network (DQN)** utilizado para resolver el ambiente `LunarLander-v3`.

El agente deberá aprender una función de valor Q que permita seleccionar la acción más conveniente para cada estado observado.

## 2.2 Arquitectura de la red neuronal

En esta sección se documentará la arquitectura de la red neuronal utilizada para aproximar la función Q.

Se deberán especificar:

- Dimensión de entrada.
- Número de capas ocultas.
- Número de neuronas por capa.
- Funciones de activación.
- Dimensión de salida.
- Justificación de la arquitectura seleccionada.

La red deberá recibir las **8 variables de observación** del ambiente y generar **4 valores Q**, correspondientes a las cuatro acciones disponibles.

## 2.3 Replay Buffer

En esta sección se documentará la implementación de la memoria de experiencias utilizada para almacenar las transiciones obtenidas durante la interacción con el ambiente.

Se deberá explicar:

- Qué información contiene cada experiencia.
- Cómo se almacenan las experiencias.
- Tamaño máximo de la memoria.
- Cómo se realiza el muestreo de los minibatches.

## 2.4 Política epsilon-greedy

En esta sección se documentará la estrategia utilizada para equilibrar la exploración y explotación durante el entrenamiento del agente.

Se deberán explicar:

- Epsilon inicial.
- Epsilon mínimo.
- Estrategia de decaimiento.
- Selección aleatoria de acciones durante la exploración.
- Selección de la acción con mayor valor Q durante la explotación.

## 2.5 Target Network y actualización de la función Q

En esta sección se documentará el uso de la red objetivo y el procedimiento utilizado para calcular los valores objetivo de la función Q.

Se deberá explicar:

- Función de pérdida utilizada.
- Cálculo del objetivo de Bellman.
- Optimización de la red principal.
- Frecuencia de actualización de la Target Network.

## 2.6 Flujo general del DQN

Se deberá incluir una explicación o diagrama del flujo general del algoritmo:

~~~
Estado
   ↓
Selección de acción (epsilon-greedy)
   ↓
Interacción con el ambiente
   ↓
Recompensa + siguiente estado
   ↓
Replay Buffer
   ↓
Muestreo de experiencias
   ↓
Cálculo del objetivo de Bellman
   ↓
Actualización de la red Q
   ↓
Actualización de Target Network
   ↓
Siguiente interacción
~~~

---

# 3. Entrenamiento y ajuste de hiperparámetros

> **Esta sección corresponde a la Persona 3.**

## 3.1 Proceso de entrenamiento

En esta sección se documentará el proceso utilizado para entrenar el agente DQN en `LunarLander-v3`.

Se deberá indicar:

- Número de episodios.
- Criterio utilizado para finalizar cada episodio.
- Estrategia de exploración.
- Frecuencia de actualización de la red objetivo.
- Método de optimización.
- Función de pérdida.

## 3.2 Hiperparámetros

Se deberán documentar los principales hiperparámetros utilizados:

| Hiperparámetro | Valor |
|---|---:|
| Número de episodios | Pendiente |
| Learning rate | Pendiente |
| Gamma (γ) | Pendiente |
| Epsilon inicial | Pendiente |
| Epsilon mínimo | Pendiente |
| Decaimiento de epsilon | Pendiente |
| Tamaño del Replay Buffer | Pendiente |
| Batch size | Pendiente |
| Frecuencia de actualización de Target Network | Pendiente |

## 3.3 Ajuste de hiperparámetros

Se documentarán las pruebas realizadas para seleccionar los hiperparámetros finales.

Para cada modificación se deberá indicar:

- Valor utilizado.
- Resultado obtenido.
- Razón para mantener o descartar la configuración.
- Impacto observado sobre el aprendizaje.

---

# 4. Resultados y evaluación

> **Esta sección corresponde a la Persona 4.**

## 4.1 Resultados del entrenamiento

En esta sección se presentarán los resultados obtenidos durante el entrenamiento del agente.

Se deberán incluir:

- Recompensa acumulada por episodio.
- Recompensa promedio.
- Desviación estándar.
- Número total de episodios.
- Mejor recompensa obtenida.
- Evolución de la recompensa durante el entrenamiento.

## 4.2 Curva de aprendizaje

Se deberá incluir una gráfica que muestre la evolución de la recompensa durante el entrenamiento.

La gráfica permitirá analizar si el agente presenta una tendencia de aprendizaje y si las recompensas mejoran a medida que aumenta el número de episodios.

## 4.3 Evaluación del agente

La evaluación deberá realizarse utilizando episodios independientes de los utilizados durante el entrenamiento.

Se deberán presentar métricas como:

- Recompensa promedio.
- Desviación estándar.
- Número de aterrizajes exitosos.
- Porcentaje de éxito.
- Comportamiento observado durante la evaluación.

## 4.4 Análisis de resultados

Se deberá analizar si el agente logró aprender una política efectiva para resolver el ambiente.

También se deberá comparar el desempeño obtenido con el objetivo esperado para `LunarLander-v3`.

---

# 5. Reflexión sobre los resultados

> **Esta sección corresponde al equipo.**

A partir de los resultados obtenidos se deberá realizar una reflexión sobre el comportamiento del agente y el proceso de aprendizaje.

Se deberán analizar aspectos como:

- ¿El agente logró aprender a aterrizar?
- ¿Cómo evolucionó la recompensa durante el entrenamiento?
- ¿El aprendizaje fue estable?
- ¿Qué hiperparámetros tuvieron mayor influencia?
- ¿Qué comportamientos presentó el agente?
- ¿Qué limitaciones se identificaron?
- ¿Qué mejoras podrían realizarse?

La reflexión deberá relacionar los resultados obtenidos con las decisiones tomadas durante la implementación y el entrenamiento.

---

# 6. Dificultades encontradas

> **Esta sección corresponde al equipo.**

Durante el desarrollo del proyecto se documentarán las principales dificultades encontradas.

Entre ellas pueden incluirse:

- Configuración e instalación del ambiente.
- Exploración y comprensión del espacio de observaciones y acciones.
- Preprocesamiento de las observaciones.
- Implementación del Replay Buffer.
- Implementación de la política epsilon-greedy.
- Implementación de la Target Network.
- Cálculo del objetivo de Bellman.
- Entrenamiento de la red neuronal.
- Ajuste de hiperparámetros.
- Evaluación del agente.
- Interpretación de los resultados.

Para cada dificultad se describirán las estrategias utilizadas para resolverla y los aprendizajes obtenidos durante el proceso.

---

# 7. Conclusiones generales

> **Esta sección corresponde a la última persona que integre el README y al equipo.**

Las conclusiones generales deberán integrar los principales resultados obtenidos durante todo el proyecto.

Se deberá analizar:

- Cumplimiento del objetivo planteado.
- Desempeño final del agente DQN.
- Capacidad del agente para resolver `LunarLander-v3`.
- Principales aprendizajes obtenidos.
- Limitaciones del enfoque.
- Posibles mejoras y trabajos futuros.

La conclusión deberá relacionar la exploración del ambiente, la implementación del DQN, el entrenamiento y los resultados de evaluación.

---

# 8. Referencias

- Gymnasium. `LunarLander-v3`: documentación oficial del ambiente.
- PyTorch. Documentación y tutoriales relacionados con Deep Q-Networks.
