# MachineEndes

Proyecto de Machine Learning para análisis de salud basado en `base.sav`. Incluye:
- Predicción de presión sistólica (`pasistolica`) con modelos de regresión.
- Clasificación de HTA combinada (`HTAcomb`).
- Clasificación de diabetes (`diabetes`).
- Panel interactivo con Streamlit.
- Visualizaciones profesionales: tablas comparativas, matrices de confusion, ROC y graficos de regresion.

## Estructura del repositorio
- `MachineEndes.ipynb`: notebook original (Jupyter en VS Code).
- `base.sav`: dataset en formato SPSS.
- `app.py`: aplicación Streamlit.
- `requirements.txt`: dependencias.
- `.gitignore`: archivos ignorados por Git.
- `runtime.txt`: versión de Python recomendada para despliegue.

## Requisitos
- Python 3.11
- Dependencias listadas en `requirements.txt`

## Uso local (Jupyter)
Abre `MachineEndes.ipynb` en VS Code y ejecuta las celdas.

## Uso local (Streamlit)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Modelos guardados (joblib)
Si activas la opcion de guardado en la app, los modelos se guardan en:
- `models/modelo_regresion.joblib`
- `models/modelo_clasificacion_hta.joblib`
- `models/modelo_clasificacion_diabetes.joblib`

## Metodologia profesional (resumen)
- Particion train/test con `random_state=42`.
- Preprocesamiento con escalado y OneHot.
- Comparacion de modelos con metricas estandar.
- Balanceo con SMOTE para HTAcomb.

## Despliegue en Streamlit Cloud
1. Sube este repositorio a GitHub.
2. En Streamlit Cloud, crea una nueva app y apunta al repo.
3. Usa `app.py` como archivo principal.

## Notas
La app usa por defecto `base.sav` en el mismo directorio. También permite cargar un archivo `.sav` desde la interfaz.
