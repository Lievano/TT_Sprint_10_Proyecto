# Modelo de Riesgo de Churn — Beta Bank

## Resumen Ejecutivo

Este proyecto construye un pipeline de machine learning para identificar clientes bancarios con alta probabilidad de abandono. Está planteado como un sistema de priorización de riesgo: el objetivo no es solo clasificar clientes, sino ayudar a un equipo de retención a decidir dónde concentrar primero sus esfuerzos.

El flujo compara distintas estrategias de modelado, maneja el desbalance de clases, optimiza el umbral de decisión para F1-score y evalúa el desempeño final en un conjunto de prueba separado.

## Problema de Negocio

El abandono de clientes aumenta los costos de adquisición, reduce el valor de vida del cliente y debilita la estabilidad de ingresos. Un equipo de retención necesita identificar clientes con riesgo antes de que la relación se pierda.

El modelo apoya esta decisión ordenando clientes según su probabilidad de churn.

## Objetivo Técnico

Predecir la variable objetivo `Exited`:

- `0` = el cliente permaneció
- `1` = el cliente abandonó

La métrica principal es F1-score porque el dataset está desbalanceado y tanto los falsos positivos como los falsos negativos importan. También se reporta AUC-ROC para evaluar la capacidad de ordenamiento del modelo.

## Dataset

El dataset contiene información bancaria a nivel cliente:

| Variable | Significado |
|---|---|
| `CreditScore` | Puntaje crediticio |
| `Geography` | País de residencia |
| `Gender` | Género |
| `Age` | Edad |
| `Tenure` | Años con el banco |
| `Balance` | Saldo en cuenta |
| `NumOfProducts` | Número de productos contratados |
| `HasCrCard` | Si tiene tarjeta de crédito |
| `IsActiveMember` | Si el cliente está activo |
| `EstimatedSalary` | Salario estimado |
| `Exited` | Variable objetivo |

Las columnas identificadoras como `RowNumber`, `CustomerId` y `Surname` se eliminan porque no aportan señal predictiva útil.

## Metodología

El proyecto sigue un flujo reproducible:

1. Carga e inspección del dataset.
2. Eliminación de identificadores no predictivos.
3. Separación de variables numéricas y categóricas.
4. Preprocesamiento mediante pipelines de scikit-learn.
5. División en train, validación y test.
6. Comparación de estrategias para manejar desbalance.
7. Ajuste de modelos seleccionados.
8. Optimización del umbral de clasificación.
9. Evaluación final en test.

## Modelado

Se evalúan Regresión Logística y Random Forest con tres estrategias de desbalance:

| Estrategia | Propósito |
|---|---|
| `class_weight` | Penaliza más los errores sobre la clase minoritaria |
| `SMOTE-NC` | Genera muestras sintéticas para datos mixtos |
| `RandomUnderSampler` | Reduce el dominio de la clase mayoritaria |

Random Forest resulta la familia más fuerte porque captura interacciones no lineales entre atributos del cliente.

## Validación

La división de datos se hace de forma estratificada para conservar la proporción de churn.

El modelo final se selecciona con base en F1-score de validación. En lugar de usar el umbral estándar de `0.5`, el proyecto usa la curva precisión-recall para encontrar el umbral que maximiza F1.

## Resultados

| Métrica | Resultado aproximado |
|---|---|
| F1-score en test | ~0.63-0.67 |
| AUC-ROC | ~0.85 |
| F1 mínimo objetivo | 0.59 |
| Familia más fuerte | Random Forest con manejo de desbalance |

El modelo final supera el objetivo mínimo de F1.

## Hallazgos

Las señales más fuertes de churn suelen incluir:

- edad
- estado de actividad
- número de productos contratados
- país, especialmente el segmento Germany
- comportamiento asociado al saldo

Estos patrones sugieren que el riesgo de abandono es conductual y de uso de producto, no solamente demográfico.

## Impacto

Este modelo puede apoyar:

- campañas de retención focalizadas
- niveles de riesgo por cliente
- flujos proactivos de contacto
- reducción de gasto innecesario en retención
- monitoreo periódico de riesgo de churn

La salida del modelo debe tratarse como una capa de apoyo a decisiones, no como una decisión final automática.

## Estructura del Repositorio

```text
customer-churn-risk-model/
├── data/
│   └── Churn.csv
├── notebooks/
│   ├── project_EN.ipynb
│   └── project_ES.ipynb
├── src/
│   └── pipeline.py
├── reports/
│   └── figures/
├── README.md
├── README_EN.md
├── README_ES.md
├── docs/
│   ├── INTERVIEW_EN.md
│   ├── INTERVIEW_ES.md
│   ├── results_summary.md
│   ├── notes.md
│   └── figures/
├── requirements.txt
└── .gitignore
```

Importante: `docs/` existe localmente para preparación de entrevista y notas extendidas, pero está ignorado por git.

## Cómo Ejecutar

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Abrir el notebook en inglés:

```bash
jupyter notebook notebooks/project_EN.ipynb
```

O abrir el notebook en español:

```bash
jupyter notebook notebooks/project_ES.ipynb
```

## Próximos Pasos

1. Crear variables conductuales, como `Balance / EstimatedSalary`.
2. Probar modelos de gradient boosting.
3. Calibrar probabilidades antes de usar scores de riesgo en operación.
4. Monitorear drift en tasa de churn y distribución de variables.
5. Empaquetar el pipeline final para scoring periódico.
