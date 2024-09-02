# Objetivo
Desarrollar un script en Python que implemente un modelo de Regresión Lineal utilizando
Keras, basado en un conjunto de datos que incluye altura y peso de personas. El modelo
debe permitir encontrar la relación entre la altura y el peso, realizar predicciones y visualizar
los resultados.

# Modelo de Regresión Lineal con Keras

Este proyecto entrena un modelo de regresión lineal utilizando Keras para predecir el peso de una persona basado en su altura. Los datos son leídos desde un archivo CSV y el modelo se entrena usando el algoritmo de Gradiente Descendente Estocástico (SGD).

# Lectura de los Datos:
Utilizar la librería Pandas para leer un archivo CSV que contiene los datos de altura y peso de
50 personas.
Almacenar estos datos en un DataFrame de Pandas y crear dos variables, x y y, donde x
almacenará la altura y del peso.

## Requisitos

Asegúrate de tener instaladas las siguientes librerías en un entorno virtual de Python:

- `pandas`
- `numpy`
- `matplotlib`
- `keras`

Creacion del entorno virutal:

```bash
py -m venv env 
```

Entrar en el entorno virutal:

```bash
.\env\Scripts\activate 
```

Instalar las librerías utilizando el siguiente comando dentro del entorno virutal:

```bash
pip install -r requirements.txt
```

# Utilizacion:
Para probar la prediccion del peso de las personas en base a su altura puede hacerlo cambiando la siguiente linea N°53

### x_pred = np.array([170])

se debe cambiar el numero dentro del parametro, siendo el mismo el valor de la altura de una persona en cm.
Por ultimo para utilizar el script se debe ejecutar el siguiente comando
```bash
py main.py
```
