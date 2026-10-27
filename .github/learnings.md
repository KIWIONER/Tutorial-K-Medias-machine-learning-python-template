# 📚 Resumen de Aprendizajes: Sistema de Agrupación y Clasificación (California Housing)

Este ejercicio representa uno de los flujos de trabajo más potentes y comunes en la industria del **Machine Learning**: la combinación de **aprendizaje no supervisado** para el descubrimiento de patrones y etiquetado automático, seguido de **aprendizaje supervisado** para la inferencia rápida y predictiva.

---

## 1. 🔄 El Patrón Híbrido: Del Aprendizaje No Supervisado al Supervisado
* **El dilema de los datos reales:** En la industria, más del 80% de los datos disponibles en bruto no vienen etiquetados (no cuentan con una variable objetivo $y$ predefinida).
* **Solución elegante:** Usar un algoritmo de clustering (**K-Means**) para descubrir patrones naturales en los datos y **generar una etiqueta automática** (`cluster`).
* **Entrenamiento supervisado posterior:** Una vez generada la etiqueta, podemos entrenar un clasificador supervisado (**Random Forest**) para predecir rápida y determinantemente a qué segmento pertenece cualquier nuevo registro que llegue en el futuro.

---

## 2. 📍 Dominio del Algoritmo K-Means
* **Agrupamiento multidimensional:** K-Means no solo agrupó casas por su geografía (`Latitude`, `Longitude`), sino que incorporó el ingreso medio (`MedInc`) como una tercera coordenada euclidiana.
* **Interpretación de negocio:** Los clusters no son simples números (0 al 5), sino **perfiles socioeconómicos reales**:
  * Zonas del sur de California (área metropolitana de Los Ángeles) divididas en poder adquisitivo bajo, medio y alto.
  * Zonas del norte (Bay Area vs. Valle Central) divididas por ingresos.
  * Un cluster específico para valores atípicos (*outliers*) de ingresos extremadamente altos.
* **Inferencia vs. Ajuste:** Se aplicó rigurosamente la diferencia entre `fit_predict()` (ajustar centroides y etiquetar Train) y `predict()` (asignar nuevos puntos de Test al centroide más cercano sin recalcular los centros).

---

## 3. 📊 Validación y Visualización de Datos
* **Inspección visual superpuesta:** Al graficar los puntos de Test sobre los de Train con bordes destacados, comprobamos visualmente que el modelo asigna coherentemente los nuevos datos en las mismas regiones geográficas.
* **Comprobación estadística:** Comparar las medias de cada variable por cluster entre Train y Test garantiza matemáticamente que el modelo generaliza sin sesgos distributivos.

---

## 4. 🌲 Clasificación Supervisada y Métricas
* **Idoneidad de los árboles de decisión:** Algoritmos como *Random Forest* son perfectos para este caso porque particionan el espacio de características en cortes perpendiculares (fronteras hiperrectangulares), emulando con casi total perfección las fronteras de K-Means.
* **Comprensión del alto rendimiento (99.54% Accuracy):** El clasificador supervisado no estaba compitiendo contra "ruido humano", sino aprendiendo una función matemática determinista generada por K-Means en el mismo espacio 3D.
* **Evaluación profesional:** Uso de la **Matriz de Confusión** (diagonal casi pura) y el **Classification Report** (Precision, Recall y F1-Score balanceados entre 0.98 y 1.00 en todas las clases).

---

## 5. 🛠️ Buenas Prácticas de Ingeniería de Software (MLOps Básico)
* **Aislamiento de entornos:** Uso de **Python 3.12** en un entorno virtual (`.venv`) con dependencias gestionadas limpiamente.
* **Persistencia de modelos:** Guardado en disco con `joblib` (`.joblib`), permitiendo cargar los modelos en memoria y realizar inferencias instantáneas en producción sin necesidad de reentrenar.
* **Del Notebook al Script Productivo:** Pasar del análisis exploratorio en `src/explore.ipynb` a un script modular y ejecutable de principio a fin en `src/app.py`.
* **Control de versiones profesional:** Configuración estricta de `.gitignore` (evitando subir binarios pesados o temporales), commits semánticos y sincronización continua con GitHub.
