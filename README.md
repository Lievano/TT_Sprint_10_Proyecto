# Proyecto Sprint 10 - Predicción de Churn (Beta Bank)

## Descripción general

Este proyecto desarrolla un **modelo predictivo de abandono de clientes (churn)** para el banco ficticio *Beta Bank*.  
El objetivo es identificar qué clientes tienen mayor probabilidad de cerrar su cuenta, de modo que el área de retención pueda actuar preventivamente.

El dataset contiene información demográfica, financiera y de comportamiento de más de 10 000 clientes, incluyendo variables como edad, país, saldo, número de productos contratados y si es un cliente activo o no.

---

## Objetivo del proyecto

- Maximizar el **F1-score** en el conjunto de prueba (`test`), con un mínimo aceptable de **0.59**.  
- Reportar también **AUC-ROC** para evaluar la capacidad global de discriminación del modelo.  
- Tratar el **desbalance de clases**, ya que los clientes que abandonan son minoría.

---

## Enfoques utilizados

Para abordar el problema, se implementaron tres estrategias de balanceo:

| Enfoque | Descripción | Modelos aplicados |
|----------|--------------|------------------|
| **class_weight** | Ajusta pesos inversos a la frecuencia de cada clase | Logistic Regression, Random Forest |
| **SMOTE-NC** | Sobremuestreo sintético para datos mixtos (numéricos y categóricos) | Logistic Regression, Random Forest |
| **RandomUnderSampler** | Submuestreo de la clase mayoritaria | Logistic Regression, Random Forest |

---

## Tecnologías y librerías

- **Python 3.12**  
- **Pandas**, **NumPy** — análisis y manipulación de datos  
- **Matplotlib**, **Seaborn** — visualización  
- **Scikit-Learn**, **Imbalanced-Learn** — modelado y balanceo  
- **SciPy** — espacios de búsqueda aleatoria (`RandomizedSearchCV`)  
- **Jupytext / VS Code Notebooks** — desarrollo reproducible en formato `.py` / `.ipynb`

---

## Estructura del proyecto

Proyecto_Sprint10_Churn/
├── Proyecto_Sprint10_Churn.py # Script principal (compatible con Jupyter Notebook)
├── Proyecto_Sprint10_Churn.ipynb # Notebook 
├── data/
│ └── Churn.csv # Dataset original
├── README.md # Descripción del proyecto
└── requirements.txt # Librerías necesarias 


---

## Flujo del análisis

1. **Carga y exploración del dataset**
   - Identificación de columnas numéricas y categóricas.
   - Análisis de valores nulos y rangos.
   - EDA visual: distribuciones, correlaciones, proporción de churn.

2. **Preprocesamiento**
   - Imputación de valores faltantes.
   - Escalado de variables numéricas.
   - Codificación categórica (One-Hot y Ordinal según el caso).

3. **Corrección del desbalance**
   - `class_weight`, `SMOTE-NC` y `RandomUnderSampler`.

4. **Entrenamiento y validación**
   - Modelos: Logistic Regression, Random Forest.
   - Optimización de umbral para F1.
   - Comparación de métricas en validación (F1, recall, AUC).

5. **Búsqueda de hiperparámetros**
   - `RandomizedSearchCV` sobre Random Forest y Logistic Regression.

6. **Evaluación final**
   - Reentrenamiento en `train + valid`.
   - Reajuste de umbral y evaluación en `test`.
   - Reporte: F1, AUC, matriz de confusión, curva ROC.

7. **Importancia de variables**
   - Permutation Importance (RF) o coeficientes (LR).

---

## Resultados clave

| Métrica | Validación | Test |
|----------|-------------|------|
| **F1-score** | ~0.65 | 0.63–0.67 |
| **AUC-ROC** | ~0.85 | ~0.85 |
| **Recall** | 0.63–0.70 | 0.62–0.68 |

**Modelo ganador:** `RandomForestClassifier` con `class_weight='balanced_subsample'`.

### Variables más influyentes
- Age  
- IsActiveMember  
- NumOfProducts  
- Geography_Germany

---

## Conclusiones

- El modelo cumple el objetivo del brief (F1 ≥ 0.59).  
- RandomForest + class_weight ofrece el mejor equilibrio entre precisión y recall.  
- El pipeline es reproducible, modular y fácilmente desplegable.  
- La optimización del umbral de decisión mejora significativamente el rendimiento frente al valor fijo (0.5).

---

## Próximos pasos

1. **Feature engineering:** crear variables derivadas (por ejemplo, ratio Balance/Salary, antigüedad normalizada).  
2. **Modelos avanzados:** probar XGBoost o LightGBM con validación cruzada estratificada.  
3. **Ajuste según estrategia de negocio:** optimizar recall o precisión según el costo de retener o ignorar un cliente.  
4. **Despliegue:** exportar pipeline con `joblib` y usarlo en scoring periódico de clientes.

---

## Autor

**Luis Liévano**  
Proyecto desarrollado como parte del **Sprint 10 — TripleTen Data Science Bootcamp (Predicción de Churn)** (2025).

