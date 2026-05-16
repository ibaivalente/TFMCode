# imports
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
# funciones
from protocolo_evaluacion import (PerformanceEvaluator, run_benchmarked_inference)

def tune_knn_baseline(dataset_name, x_train, y_train, x_test, y_test, iterations):
    """
    Realiza una búsqueda de los mejores hiperparámetros para el baseline k-NN.
    y_train/y_test deben contener todas las columnas necesarias (x, y, [piso, edificio])
    """
    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    metrics = ['euclidean', 'manhattan']

    best_results = None
    best_mae = float('inf')
    all_logs = []

    print(f"Búsqueda rejilla para el dataset: {dataset_name}")

    for metric in metrics:
        for k in k_values:
            # instanciamos el modelo kNN
            knn = KNeighborsRegressor(n_neighbors=k, metric=metric, weights='uniform')

            # ejecutamos la inferencia con el número de iteraciones indicado
            # llamamos a la función importada de utils
            # retorna las predicciones, el tiempo medio y el tiempo estándar de inferencia
            y_pred, avg_time, std_time = run_benchmarked_inference(knn.fit(x_train, y_train), x_test, iterations=iterations)

            # instanciamos un objeto de la clase evaluador importada de utils
            evaluator = PerformanceEvaluator(dataset_name)

            if dataset_name == "UJI1":
                # forzamos la conversión a array de np para poder hacer el slicing
                y_test_arr = y_test.values if isinstance(y_test, pd.DataFrame) else np.asarray(y_test)
                y_pred_arr = np.asarray(y_pred)
                # recordamos que nuestro target se compone de: x, y, z, piso, edificio
                # en UJI1 los índices 0 y 1 son x e y; el índice 3 es el piso; el índice 4 es el edificio
                # recordamos que los slicing funcionan como el primer valor incluido, el segundo valor excluido
                # y que el índice inicial es el 0
                # redondeamos
                y_pred_labels = np.round(y_pred_arr[:, [4, 3]])
                # llamamos al método del objeto
                # para UJI1 retorna accuracy de piso y edificio, MAE en 2D, percentiles 75 y 95
                res = evaluator.calculate_precision(y_test_arr[:, 0:2], y_pred_arr[:, 0:2],
                                                   building_floor_true=y_test_arr[:, [4, 3]],
                                                   building_floor_pred=y_pred_labels)
            else:
                # para MAN1 retorna MAE en 2D, percentiles 75 y 95
                res = evaluator.calculate_precision(y_test, y_pred)

            # registra los resultados de eficiencia
            # añade a la lista de logs toda la información
            res['k'] = k
            res['metric'] = metric
            res['MTQ_ms'] = (avg_time / len(x_test)) * 1000

            all_logs.append(res)

            # criterio de selección del mejor modelo: el menor MAE (en UJI1, MAE cuando se acierta edificio/piso)
            if res['MAE_2D'] < best_mae:
                best_mae = res['MAE_2D']
                best_results = res

            # imprime información de la iteración en curso
            print(f"Prueba: k={k}, métrica={metric} --> MAE: {res['MAE_2D']:.2f}m")

    # retorna la información del método de cálculo de precisión en forma de df y los resultados del mejor modelo
    return pd.DataFrame(all_logs), best_results