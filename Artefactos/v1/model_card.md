# Model Card — MLP
**Versión:** v1
**Entorno:** Python 3.12.12 | scikit-learn 1.6.1

## Datos
Archivo: `cardio_data_clean.csv`
Shape: (70000, 16)
Objetivo: `cardio` (Sin_riesgo=0, Con_riesgo=1)
Prevalencia (Con_riesgo=1) — TRAIN: 0.500 | TEST: 0.500

## Entrenamiento
Split 80/20 estratificado (random_state=42).
Preprocesamiento: StandardScaler(num) + OneHotEncoder(cat) + SMOTE para balancear clases.

## Modelo seleccionado
**MLP**
Umbral óptimo de decisión (Paso 8): **0.50**

## Métricas en TEST
ACC=0.721 | BALACC=0.721 | PREC=0.740 | REC=0.683 | F1=0.710 | MCC=0.444
ROC-AUC=0.786 | PR-AUC=0.772

## Artefactos generados
- `pipeline_MLP.joblib`
- `input_schema.json`
- `label_map.json`
- `decision_policy.json`
