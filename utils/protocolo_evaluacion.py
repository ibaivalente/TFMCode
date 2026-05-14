# imports
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import metrics

# clase evaluador
class PerformanceEvaluator:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.results = {}

    def calculate_precision(self, y_true, y_pred, building_floor_true=None, building_floor_pred=None):
        """
        y_true/y_pred: coordenadas [x, y]
        building_floor: [id_edificio, id_piso] solo para el caso de UJI1 multiedificio y multiplanta
        """
        # cálculo del error de distancia euclídea en 2D
        errors = np.sqrt(np.sum((y_true - y_pred)**2, axis=1))
        
        # aplicación de la lógica jerárquica para UJI1
        if self.dataset_name == "UJI1":
            # calculamos aciertos de clasificación
            b_hit = (building_floor_true[:, 0] == building_floor_pred[:, 0])
            f_hit = (building_floor_true[:, 1] == building_floor_pred[:, 1])
            hit_rate = np.logical_and(b_hit, f_hit)
            
            accuracy_b = np.mean(b_hit) * 100
            accuracy_f = np.mean(f_hit) * 100
            
            # MAE solo si el edificio y piso son correctos
            mae_2d = np.mean(errors[hit_rate]) if any(hit_rate) else np.nan
            self.results['Building_Accuracy'] = accuracy_b
            self.results['Floor_Accuracy'] = accuracy_f
        else:
            mae_2d = np.mean(errors)

        # calculamos percentiles 75 y 95 y MAE general
        self.results['MAE_2D'] = mae_2d
        self.results['P75'] = np.percentile(errors, 75)
        self.results['P95'] = np.percentile(errors, 95)
        # guardamos errores para la visualización del CDF
        self.raw_errors = errors
        
        return self.results

    def plot_cdf(self):
        """
        Genera la visualización de la CDF del error
        """
        sorted_errors = np.sort(self.raw_errors)
        cdf = np.arange(len(sorted_errors)) / float(len(sorted_errors))
        plt.plot(sorted_errors, cdf, label=f"CDF {self.dataset_name}")
        plt.xlabel("Error en metros")
        plt.ylabel("Probabilidad acumulada")
        plt.grid(True)
        plt.show()

    def calculate_efficiency(self, start_time, end_time, n_queries, original_size, reduced_size):
        """
        Calcula métricas de tiempo y alivio
        """
        total_time = end_time - start_time
        self.results['MTQ_ms'] = (total_time / n_queries) * 1000
        self.results['Reduction_Factor'] = (1 - (reduced_size / original_size)) * 100
        return self.results
    
# función benchmark de inferencia
# editar número iteraciones como sea necesario
def run_benchmarked_inference(model, test_data, iterations):
    """
    Ejecuta la inferencia múltiples veces para obtener tiempos estables.
    """
    execution_times = []
    predictions = None
    
    print(f"Iniciando benchmark: {iterations} iteraciones...")
    
    for i in range(iterations):
        t0 = time.time()
        # realizamos la predicción
        current_preds = model.predict(test_data)
        t1 = time.time()
        
        execution_times.append(t1 - t0)
        
        # guardamos las predicciones de la última vuelta (o la primera, da igual si es determinista)
        if i == 0:
            predictions = current_preds
            
    avg_time = np.mean(execution_times)
    std_time = np.std(execution_times)
    
    print(f"Tiempo medio: {avg_time*1000:.2f} ms, desviación: {std_time*1000:.2f} ms")
    
    return predictions, avg_time, std_time