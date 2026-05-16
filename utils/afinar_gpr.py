# imports
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.neighbors import KNeighborsRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning
# silenciamos los warnings de convergencia porque hemos visto que el modelo ya ha alcanzado su límite de rendimiento y no mejora aunque cambiemos los parámetros
warnings.filterwarnings('ignore', category=ConvergenceWarning)
# funciones
from protocolo_evaluacion import (PerformanceEvaluator, run_benchmarked_inference)

# función para aplicar el suavizado espacial
def apply_spatial_smoothing(train_rss, train_crd):
    """
    Agrupa las huellas por coordenadas únicas, calcula la media para obtener puntos limpios
    """
    # creamos copias locales para proteger los df originales
    rss_df = train_rss.copy()
    crd_df = train_crd.copy()

    #renombramos las coordenadas a strings únicos
    crd_df.columns = [f'COORD_{i}' for i in range(crd_df.shape[1])]
    coord_cols = list(crd_df.columns)

    # combinamos temporalmente para agrupar de forma unificada
    full_train = pd.concat([crd_df, rss_df], axis=1)

    # agrupamos por coordenada y sacamos la media
    smoothed_df = full_train.groupby(coord_cols).mean().reset_index()

    x_smoothed = smoothed_df[coord_cols].values
    y_smoothed = smoothed_df.drop(columns=coord_cols).values

    print(f"Suavizado hecho. Longitud inicial {len(train_rss)}, longitud final {len(x_smoothed)}")
    return x_smoothed, y_smoothed

# función para poner en marcha el pipeline que afina el GPR
def tune_gpr_pipeline(x_train_raw, y_train_raw, x_test, y_test, best_knn_k, best_knn_metric, iterations):
    """
    Realiza búsqueda de rejilla sobre los hiperparámetros del GPR, genera el mapa sintético, evalúa el rendimiento final
    """
    # llamamos a la función de suavizado pasándole RSS y coordenadas cruzas y retornando los df suavizados
    x_train_space, y_train_space = apply_spatial_smoothing(y_train_raw, x_train_raw)

    # definimos de la rejilla de parámetros
    candidate_kernels = {
        # con length_scale y length_scale_bounds indicamos el radio de influencia de la medición. Indicamos el rango 1e-1 a 1e5 para abarcar de pocos centrímetros a muchos metros
        # el noise_level de 0,1 absorbe de forma estadística el ruido y las fluctuaciones de las 110 capturas, actúa como colchón para que el GPR no sobreajuste
        'RBF': RBF(length_scale=10.0, length_scale_bounds=(1e-1, 1e5)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-7, 1e2)),
        # para la definción del kernel Matérn indicamos el valor de nu 0,5 (1/2), 1,5 (3/2) y 2,5 (5/2), que son los valores más habituales
        'Matern_0.5': Matern(length_scale=10.0, nu=0.5, length_scale_bounds=(1e-1, 1e5)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-7, 1e2)),
        'Matern_1.5': Matern(length_scale=10.0, nu=1.5, length_scale_bounds=(1e-1, 1e5)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-7, 1e2)),
        'Matern_2.5': Matern(length_scale=10.0, nu=2.5, length_scale_bounds=(1e-1, 1e5)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-7, 1e2))
    }
    # con normalize true indicamos media 0 y con false indicamos media constante, donde el optimizador encuentra el valor base más adecuado para el mapa
    normalize_options = [True, False]

    # inicializamos variables para empezar las iteraciones
    best_mae = float('inf')
    best_gpr_results = None
    best_evaluator = None
    gpr_logs = []

    # con shape[1] sobre y_train_space accedemos al valor de APs, es decir, 28 para MAN1
    num_aps = y_train_space.shape[1]

    # iteramos sobre los kernels y sobre las opcines de normalización
    for kernel_name, base_kernel in candidate_kernels.items():
        for norm_y in normalize_options:
            print(f"\nEvaluando Configuración GPR con kernel {kernel_name} y normalización {norm_y}")

            # lista para almacenar los 28 modelos GPR de esta iteración
            trained_gps = []

            # entrenamos un GPR independiente por cada AP utilizando los 130 puntos suavizados
            # instanciamos un objeto regresor de la clase GaussianProcessRegressor con los parámetros de la iteración en curso y añadimos el modelo entrenado a la lista
            for ap_idx in range(num_aps):
                gp = GaussianProcessRegressor(kernel=base_kernel, normalize_y=norm_y, random_state=1)
                gp.fit(x_train_space, y_train_space[:, ap_idx])
                trained_gps.append(gp)

            # creamos el mapa de radio sintético
            # reconstruimos la matriz de señales estimadas sobre las 130 posiciones reales
            synthetic_rss_train = np.zeros_like(y_train_space)
            for ap_idx in range(num_aps):
                synthetic_rss_train[:, ap_idx] = trained_gps[ap_idx].predict(x_train_space)


            # configuramos el asignador k-NN fijando los mejores parámetros del baseline anterior
            knn_matcher = KNeighborsRegressor(n_neighbors=best_knn_k, metric=best_knn_metric, weights='uniform')
            # entrenamos el asignador usando el mapa sintético reducido de 130 muestras
            knn_matcher.fit(synthetic_rss_train, x_train_space)

            # perparamos los arrays para test
            # señales RSS con los 28 APs
            test_rss_input = y_test.values if hasattr(y_test, 'values') else y_test
            # coordenadas reales
            test_crd_true = x_test.values if hasattr(x_test, 'values') else x_test

            # medimos el tiempo de inferencia con el benchmark llamando a la función de los utils
            y_pred_crd, avg_time, _ = run_benchmarked_inference(
                model=knn_matcher,
                test_data=test_rss_input,
                iterations=iterations
            )

            # instanciamos un objeto evaluador de la clase PerformanceEvaluator para evaluar la precisión espacial
            evaluator = PerformanceEvaluator("MAN1")
            res = evaluator.calculate_precision(
                y_true=test_crd_true,
                y_pred=y_pred_crd
            )

            # añadimos las métricas
            res['kernel'] = kernel_name
            res['normalize_y'] = norm_y
            res['MTQ_ms'] = (avg_time / len(x_test)) * 1000
            res['Reduction_Factor'] = (1 - (len(x_train_space) / len(x_train_raw))) * 100

            gpr_logs.append(res)
            print(f"Resultado: MAE = {res['MAE_2D']:.2f}m, reducción mapa = {res['Reduction_Factor']:.2f}%")

            if res['MAE_2D'] < best_mae:
                best_mae = res['MAE_2D']
                best_gpr_results = res
                best_evaluator = evaluator

    return pd.DataFrame(gpr_logs), best_gpr_results, best_evaluator