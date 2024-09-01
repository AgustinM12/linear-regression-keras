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

# Indicar que los randoms sean siempre iguales
np.random.seed(2)

# Crear el modelo
modelo = Sequential()

# # Configuracion del modelo
input_dim = 1
output_dim = 1
modelo.add(Dense(units=output_dim, input_dim=input_dim, activation='linear'))

# # Stochastic Gradient Descent, indica que el método de entrenamiento será precisamente el del Gradiente Descendente.
sgd = SGD(learning_rate=0.0004)

# # Agregar la funcion de perdida "error cuadratico medio" y el optimizador 
modelo.compile(loss='mse', optimizer=sgd)

# # ver las caracteristicas del modelo
modelo.summary()

# # configurar las epocas, cantidad de datos que usara en el entrenamiento y el historial del entrenamiento 
num_epochs = 10000
batch_size = x.shape[0]
historia = modelo.fit(x, y, epochs=num_epochs, batch_size=batch_size,
verbose=1)

# definir las capas que tendra el modelo
# Accede a la primera capa
primer_capa = modelo.layers[0]

# Obtén los pesos y sesgos de la primera capa
w, b = primer_capa.get_weights()
print("Pesos (w):")
print(w)

print("Sesgos (b):")
print(b)

# probar el modelo de prediccion 
x_pred = np.array([170])
y_pred = modelo.predict(x_pred)
print(f"El peso de una persona con una altura {x_pred} sera aproximadamente: {y_pred}")

# graficar los datos de perdida durante el entrenamiento del modeloplt.subplot(1,2,1)
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