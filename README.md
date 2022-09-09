# Implemetaci-n_con_Framework

 Se tomó un modelo de regresión logística para poder predecir si un tumor es maligno o benigno. 
 Este modelo fue desarrollado usando scikit-learn. Se logró un valor de
 accuracy cercano al 97% incluso haciendo un cross validation, 
 lo que nos dice que el modelo no está en estado de overfit. En el documento "analisis ML"
 se explica el proceso llevado a cabo, así como los resultados obtenidos.
 
 Se usó el dataset breast-cancer-wisconsin.data que incluye varias características de los tumores a evaluar, como radio, textura, etc.
 (para más información consular el breast-cancer-wisconsin.names)
 
 En el programa regresion_logistica.py se lleva a cabo el procesamiento de los datos en dataframes, así como el proceso de entrenamiento y testeo del modelo generado,
 dicho programa te imprime en consola:
- accuracy del modelo en etapa de testeo
- score del cross validation para validar que no hay overfitting
- Matriz de confusión para ver los falsos negativos y falsos positivos
- score del cross validation para un modelo regularizado Ridge
- una predicción para un input dado (input: arreglo de 8 valores)(output: 2 o 4, dependiendo de si el tumor se predice como benigno o maligno respectivamente)

Y grafica:
- el score en la etapa de entrenamiento comparado con el score obtenido al hacer un cross validation
