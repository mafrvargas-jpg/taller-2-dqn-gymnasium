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

## 2.1 Descripción general

Se implementó un agente Deep Q-Network (DQN) para resolver el ambiente `LunarLander-v3`.

El agente utiliza una red neuronal para aproximar la función Q y seleccionar la acción con mayor valor esperado para cada estado. La implementación incorpora los componentes principales del algoritmo DQN:

- Q-Network.
- Target Network.
- Replay Buffer.
- Política epsilon-greedy.
- Actualización mediante la ecuación de Bellman.
- Optimizador Adam.

El agente recibe como entrada el vector de estado de 8 variables y genera como salida 4 valores Q, uno por cada acción disponible en el ambiente.

## 2.2 Arquitectura de la red neuronal

La arquitectura utilizada para aproximar la función Q fue:

~~~text
Entrada: 8 variables
        ↓
Capa completamente conectada: 8 → 128
        ↓
ReLU
        ↓
Capa completamente conectada: 128 → 128
        ↓
ReLU
        ↓
Capa de salida: 128 → 4
~~~

La red utiliza dos capas ocultas de 128 neuronas con funciones de activación ReLU. La capa de salida contiene 4 neuronas, correspondientes a las cuatro acciones disponibles en `LunarLander-v3`.

Esta arquitectura permite trabajar con el vector de observaciones de baja dimensión del ambiente y proporciona suficiente capacidad para aproximar la función Q sin utilizar una red excesivamente compleja.

## 2.3 Replay Buffer

Se utilizó un Replay Buffer con capacidad para almacenar hasta 100.000 experiencias.

Cada experiencia contiene:

~~~text
(estado, acción, recompensa, siguiente_estado, estado_terminal)
~~~

Durante el entrenamiento, las experiencias se almacenan en el buffer y posteriormente se seleccionan lotes aleatorios de 64 experiencias para actualizar la red.

El uso de Replay Buffer permite reducir la correlación entre experiencias consecutivas y mejora la estabilidad del aprendizaje.

## 2.4 Política epsilon-greedy

La selección de acciones se realizó mediante una política epsilon-greedy.

Al inicio del entrenamiento se utilizó un valor de epsilon igual a 1.0, favoreciendo la exploración del ambiente. Durante el entrenamiento, epsilon disminuyó progresivamente hasta alcanzar un valor mínimo de 0.01.

De esta manera, el agente comenzó explorando diferentes acciones y progresivamente pasó a utilizar con mayor frecuencia las acciones que la red neuronal estimaba como más convenientes.

## 2.5 Target Network y actualización de la función Q

Se utilizaron dos redes neuronales:

- **Q-Network:** red principal que se actualiza durante el entrenamiento.
- **Target Network:** red utilizada para calcular los valores objetivo.

La Target Network se actualizó cada 10 episodios copiando los pesos de la Q-Network.

Para calcular el valor objetivo se utilizó la ecuación de Bellman:

~~~text
Q_target = recompensa + γ × max(Q_siguiente)
~~~

cuando el episodio no había terminado.

Para estados terminales, el valor futuro no se considera.

El factor de descuento utilizado fue:

~~~text
γ = 0.99
~~~

La función de pérdida se calculó comparando los valores Q estimados por la Q-Network con los valores objetivo obtenidos mediante la ecuación de Bellman.

## 2.6 Hiperparámetros utilizados

Los principales hiperparámetros utilizados fueron:

| Hiperparámetro | Valor |
|---|---:|
| Episodios de entrenamiento | 1.000 |
| Learning rate | 0.001 |
| Gamma | 0.99 |
| Batch size | 64 |
| Capacidad Replay Buffer | 100.000 |
| Actualización Target Network | Cada 10 episodios |
| Epsilon inicial | 1.0 |
| Epsilon mínimo | 0.01 |

Los valores fueron seleccionados buscando un equilibrio entre estabilidad del aprendizaje y velocidad de entrenamiento.

# 2.7. Entrenamiento

El agente fue entrenado durante **1.000 episodios**.

Durante el entrenamiento se almacenó la recompensa acumulada obtenida en cada episodio y se realizó seguimiento de la evolución del valor de epsilon.

Al inicio del entrenamiento se observaron recompensas predominantemente negativas. Posteriormente, el agente comenzó a mejorar progresivamente su desempeño, alcanzando recompensas positivas y superiores a 200 en diferentes etapas del entrenamiento.

La mejor recompensa individual obtenida durante el entrenamiento fue de **303.47**, mientras que la recompensa promedio considerando los 1.000 episodios fue de **53.26**.

La gráfica de evolución de la recompensa muestra una tendencia general de mejora, aunque con fluctuaciones importantes entre episodios.

# 2.8. Resultados y evaluación

Después del entrenamiento se realizó una evaluación independiente utilizando **100 episodios**, sin exploración aleatoria. Durante esta etapa el agente seleccionó en cada estado la acción con mayor valor Q estimado por la red.

Los resultados fueron:

| Métrica | Resultado |
|---|---:|
| Episodios evaluados | 100 |
| Recompensa promedio | **190.87** |
| Desviación estándar | **118.31** |
| Episodios con recompensa ≥ 200 | **70/100 (70%)** |
| Mejor recompensa | **290.87** |
| Peor recompensa | **-183.08** |

El agente alcanzó una recompensa igual o superior a 200 en el **70% de los episodios evaluados**.

La recompensa promedio de 190.87 se encuentra ligeramente por debajo del umbral de 200 utilizado como referencia para considerar resuelto el ambiente. Sin embargo, el porcentaje de episodios que supera dicho valor evidencia que el agente aprendió una política capaz de obtener un buen desempeño en una proporción importante de las evaluaciones.

La desviación estándar de 118.31 muestra una variabilidad considerable entre episodios. Aunque la mayoría de los resultados se encuentran en rangos altos, también se presentaron algunos episodios con recompensas bajas o negativas.

# 2.9. Reflexión sobre los resultados

Los resultados muestran que el agente logró aprender progresivamente una estrategia para controlar la nave en `LunarLander-v3`.

Una de las principales evidencias de aprendizaje es la evolución de la recompensa durante los 1.000 episodios. El entrenamiento comenzó con recompensas predominantemente negativas y posteriormente alcanzó valores positivos y superiores a 200 en diferentes etapas.

La evaluación sobre 100 episodios mostró que el agente consiguió superar el umbral de 200 en 70 episodios. Esto indica que la política aprendida es efectiva en una proporción significativa de los casos.

Sin embargo, el desempeño no fue completamente estable. La desviación estándar de 118.31 y la presencia de episodios con recompensas negativas muestran que el agente todavía puede presentar fallos durante el aterrizaje.

Entre los posibles factores se encuentran la complejidad de controlar simultáneamente la posición, velocidad, orientación y uso de los motores, así como la sensibilidad de la recompensa ante pequeñas variaciones en la trayectoria.

Como posibles mejoras futuras se podrían explorar diferentes arquitecturas de red, tasas de aprendizaje, estrategias de actualización de la Target Network y técnicas avanzadas como Double DQN o Dueling DQN.

# 2.10. Dificultades encontradas

Durante la implementación fue necesario integrar correctamente los diferentes componentes del algoritmo DQN, especialmente la Q-Network, la Target Network y el Replay Buffer.

También fue necesario ajustar el comportamiento de la política epsilon-greedy para permitir suficiente exploración al inicio del entrenamiento y aumentar progresivamente la explotación de la política aprendida.

Otra dificultad estuvo relacionada con la variabilidad propia del ambiente `LunarLander-v3`. A pesar de que el agente logró obtener recompensas superiores a 200 en numerosos episodios, también se presentaron episodios con resultados considerablemente menores.

Esto permitió identificar que un buen desempeño promedio no garantiza que el agente tenga un comportamiento completamente estable en todas las ejecuciones.

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
