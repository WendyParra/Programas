import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

# Desactivar optimizaciones de oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Función para preprocesar las imágenes
def preprocess_image(image, target_size=(64, 64)):
    if isinstance(image, str):  # Si es una ruta, carga la imagen
        image = cv2.imread(image)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer la imagen en la ruta: {image}")
    # Procesa la imagen como una matriz
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image / 255.0  # Normalizar entre 0 y 1

# Generación de datos de ejemplo
def load_dataset(image_paths, labels, target_size=(64, 64)):
    images = []
    for img_path in image_paths:
        if os.path.exists(img_path):  # Verifica si la ruta es válida
            images.append(preprocess_image(img_path, target_size))
        else:
            print(f"Ruta no válida: {img_path}")
    return np.array(images), np.array(labels[:len(images)])

# Rutas de las imágenes y etiquetas
image_paths = [

    'C:/Users/wendy/Downloads/python/Triangulo.PNG',
    'C:/Users/wendy/Downloads/python/circulo.jpg',
    'C:/Users/wendy/Downloads/python/cuadrado.png'
]
labels = [0, 1, 2]  # Etiquetas: 0-Cuadrado, 1-Círculo, 2-Triángulo

X, y = load_dataset(image_paths, labels)
if len(X) == 0:
    raise ValueError("No se cargaron imágenes válidas. Verifica las rutas.")
y = to_categorical(y, num_classes=3)  # Codificación one-hot

# Dividir datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo
model = Sequential([
    Flatten(input_shape=(64, 64, 3)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # Tres clases: cuadrado, círculo, triángulo
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)

# Guardar el modelo entrenado
model.save('shape_detector_model.h5')

# Usar el modelo para predicción
def predict_shape(image, model, target_size=(64, 64)):
    img = preprocess_image(image, target_size)
    img = np.expand_dims(img, axis=0)  # Añadir dimensión para el batch
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    return class_idx

# Cargar el modelo entrenado
model = tf.keras.models.load_model('shape_detector_model.h5')

# Procesar la imagen para detección de contornos y usar el modelo
image = cv2.imread('C:/Users/wendy/Downloads/python/FigurasColores.png')
if image is None:
    raise FileNotFoundError("No se pudo leer la imagen principal. Verifica la ruta.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 10, 150)
canny = cv2.dilate(canny, None, iterations=1)
canny = cv2.erode(canny, None, iterations=1)

# Detección de contornos
cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    roi = image[y:y+h, x:x+w]  # Región de interés
    if roi.size > 0 and roi.shape[0] > 10 and roi.shape[1] > 10:  # Verifica dimensiones mínimas
        try:
            class_idx = predict_shape(roi, model)
            label = ["Cuadrado", "Círculo", "Triángulo"][class_idx]
            cv2.putText(image, label, (x, y-5), 1, 1, (0, 255, 0), 1)
        except Exception as e:
            print(f"Error procesando ROI: {e}")
    cv2.drawContours(image, [c], 0, (0, 255, 0), 2)

# Mostrar imagen final con las etiquetas
cv2.imshow('Detección de Figuras', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
