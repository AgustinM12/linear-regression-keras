import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import SGD

# Cargar los datos del csv
datos = pd.read_csv('altura_peso.csv', sep=",")

x = datos["Altura"].values
y = datos["Peso"].values

# Normalizar los datos SOLUCIONA EL ERROR DE LOS NAN e INF
x = (x - np.mean(x)) / np.std(x)
y = (y - np.mean(y)) / np.std(y)

print(x)
print(y)

# Indicar que los randoms sean siempre iguales
np.random.seed(2)

# Crear el modelo
modelo = Sequential()

# Configuracion del modelo con inicialización HeNormal
modelo.add(Dense(units=1, input_dim=1, activation='linear'))

# Stochastic Gradient Descent con un learning rate más bajo
sgd = SGD(learning_rate=0.0004)

# Compilar el modelo
modelo.compile(loss='mse', optimizer=sgd)

# Ver las características del modelo
modelo.summary()

# Configurar las épocas y el historial del entrenamiento
batch_size = x.shape[0]

# Monitorizar el entrenamiento
# aproximadamente en 6000 epocas se estanca la perdida 
historia = modelo.fit(x, y, epochs=10000, batch_size=batch_size, verbose=1)
w, b = modelo.layers[0].get_weights()
print(f"Pesos (w) = {w[0][0]}, Sesgos (b) = {b[0]}")

# probar el modelo de prediccion 
x_pred = np.array([170])
y_pred = modelo.predict(x_pred)
print(f"El peso de una persona con una altura {x_pred[0]} sera aproximadamente: {y_pred[0][0]}")

# Graficar los datos de pérdida durante el entrenamiento del modelo
plt.subplot(1,2,1)
plt.plot(historia.history['loss'])
plt.xlabel('epoch')
plt.ylabel('ECM')
plt.title('ECM vs. epochs')
y_regr = modelo.predict(x)
plt.subplot(1, 2, 2)
plt.scatter(x,y)
plt.plot(x,y_regr,'r')
plt.title('Datos originales y regresión lineal')
plt.show()
