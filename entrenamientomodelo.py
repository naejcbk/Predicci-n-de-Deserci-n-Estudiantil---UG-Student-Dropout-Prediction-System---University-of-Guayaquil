import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# 1. CARGA DE DATOS
# Se utiliza la librería pandas para leer el archivo CSV que contiene el historial académico.
df = pd.read_csv('dataset.csv')

# 2. DEFINICIÓN DE VARIABLES (Features y Target)
# X: Son las características que el modelo usará para aprender (promedio, asistencia y semestre).
X = df[['promediohistorico', 'promedioasistencia', '# de Semestre']]
# y: Es el "objetivo" o la respuesta que queremos predecir (si desertó o no).
y = df['Desertor']

# 3. DIVISIÓN DEL DATASET
# Se separan los datos en dos grupos: uno para que el modelo estudie (train) y otro para examinarlo (test).
# test_size=0.2: El 20% de los datos se reserva para la evaluación final.
# stratify=y: Asegura que la proporción de desertores y no desertores sea igual en ambos grupos.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. CONFIGURACIÓN DEL MODELO (Bosque Aleatorio)
# Se crea el clasificador con parámetros específicos para evitar errores de predicción:
modelo = RandomForestClassifier(
    n_estimators=100,          # Crea 100 árboles de decisión para votar por el resultado.
    max_depth=5,               # Limita la profundidad de los árboles para evitar que memoricen los datos (overfitting).
    min_samples_leaf=10,       # Pide que al menos 10 casos coincidan para crear una "hoja" de decisión.
    random_state=42,           # Asegura que el entrenamiento sea reproducible y no cambie al azar.
    class_weight='balanced'    # Crucial: Da más peso a los casos de deserción para compensar si hay pocos en el CSV.
)

# 5. ENTRENAMIENTO
# El modelo busca patrones y relaciones entre las notas/asistencia y la probabilidad de abandonar.
modelo.fit(X_train, y_train)

# 6. EXPORTACIÓN DEL MODELO
# Se guarda el modelo entrenado en un archivo físico (.pkl). 
# Esto permite que la aplicación de Streamlit pueda usarlo después sin tener que volver a entrenar.
joblib.dump(modelo, 'modelo_desercion.pkl')

print("Modelo de IA entrenado y guardado como 'modelo_desercion.pkl'.")
