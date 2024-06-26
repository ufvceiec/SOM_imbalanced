# SOM desbalanceado

Este repositorio almacena los archivos necesarios para la ejecucción del proyecto SOM desbalanceado. Este proyecto combina 5 técnicas de oversampling y 5 técnicas de undersampling, y compara los resultados con un mapa SOM. Esto es así para comprobar, mediante los errores de los mapas, cuáles son las mejores combinaciones de técnicas.


- SOM.ipynb: En este notebook está el desarrollo del proyecto. Todas las partes del notebook siguen el mismo criterio: se carga el dataset, se entrena un mapa SOM con el dataset desbalanceado, se aplican las 25 técnicas al dataset y se clasifican sobre ese mapa entrenado el dataset balanceado con cada una de las técnicas.
- MLP.ipynb: Este notebook contiene los MLP con en los cuáles nos hemos apoyado para decidir cuáles de las técnicas son las mejores. Se hace un tunning de hiperparámetros y el mejor MLP se entrena con el dataset desbalanceado y con el dataset balanceado con las 25 técnicas. Cabe destacar que cada MLP es diferente, es decir, hay una búsqueda de hiperparámetros para cada MLP.
- Resultados: Esta carpeta contiene los .csv con los resultados de los MLP con el nombre de las técnicas usadas.
- src: Esta carpeta contiene los archivos varios, siendo workflow.py uno necesario para la correcta ejecución de los notebooks, ya que tiene funciones auxiliares que se llaman desde el notebook principal, como la creación de datos sintéticos.