# Imbalanced SOM

This repository stores the necessary files for the execution of the Unbalanced SOM project. This project combines 5 oversampling techniques and 5 undersampling techniques, and compares the results using a SOM (Self-Organizing Map). The goal is to determine, based on the errors of the maps, which combinations of techniques yield the best results.

## Files and Folders

- **SOM.ipynb**: This notebook contains the development of the project. All sections of the notebook follow the same structure: the dataset is loaded, a SOM is trained with the unbalanced dataset, the 25 techniques are applied to the dataset, and the balanced dataset (using each technique) is classified using the trained SOM.
- **MLP.ipynb**: This notebook contains the MLPs (Multilayer Perceptrons) used to help determine which techniques are the best. Hyperparameter tuning is performed, and the best MLP is trained with the unbalanced dataset and the dataset balanced using the 25 techniques. It is important to note that each MLP is different, meaning hyperparameter search is conducted for each MLP.
- **Resultados**: This folder contains the `.csv` files with the results of the MLPs, named according to the techniques used.
- **src**: This folder contains various files, including `workflow.py`, which is necessary for the correct execution of the notebooks. It includes helper functions called from the main notebook, such as the generation of synthetic data.

---

# SOM desbalanceado

Este repositorio almacena los archivos necesarios para la ejecucción del proyecto SOM desbalanceado. Este proyecto combina 5 técnicas de oversampling y 5 técnicas de undersampling, y compara los resultados con un mapa SOM. Esto es así para comprobar, mediante los errores de los mapas, cuáles son las mejores combinaciones de técnicas.

## Archivos y Carpetas
- **SOM.ipynb**: En este notebook está el desarrollo del proyecto. Todas las partes del notebook siguen el mismo criterio: se carga el dataset, se entrena un mapa SOM con el dataset desbalanceado, se aplican las 25 técnicas al dataset y se clasifican sobre ese mapa entrenado el dataset balanceado con cada una de las técnicas.
- **MLP.ipynb**: Este notebook contiene los MLP con en los cuáles nos hemos apoyado para decidir cuáles de las técnicas son las mejores. Se hace un tunning de hiperparámetros y el mejor MLP se entrena con el dataset desbalanceado y con el dataset balanceado con las 25 técnicas. Cabe destacar que cada MLP es diferente, es decir, hay una búsqueda de hiperparámetros para cada MLP.
- **Resultados**: Esta carpeta contiene los `.csv` con los resultados de los MLP con el nombre de las técnicas usadas.
- **src**: Esta carpeta contiene los archivos varios, siendo `workflow.py` uno necesario para la correcta ejecución de los notebooks, ya que tiene funciones auxiliares que se llaman desde el notebook principal, como la creación de datos sintéticos.
