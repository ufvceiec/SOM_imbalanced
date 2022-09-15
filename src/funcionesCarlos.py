#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
"""
NOMBRE DEL ARCHIVO: "funcionesCarlos.py"
AUTOR: CARLOS ARRANZ LUQUE
FECHA: 2022
DESCRIPCIÓN: ARCHIVO CON FUNCIONES DE PYHTON PARA EL PROYECTO DE EVALUACIÓN DE TÉCNICAS DE BALANCEO DEL CEIEC
"""
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######



#####---------------------LIBRERIAS-----------------------------------------------------------------###### 

from sklearn import preprocessing
import pandas as pd

#####---------------------FIN LIBRERIAS-------------------------------------------------------------###### 



#####---------------------FUNCIONES-----------------------------------------------------------------###### 

"""
   Normalización mediante la técnica minmax de un data frame
"""
def min_max(df):

    nombres_columnas = df.columns 
    # Guardo los nombres de las columnas para no perderlos ya que la función que usaremos sustituye los
    # nombres de las columnas por números
    
    # Usamos el MinMax que viene con SKLearn ---> from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    x=df.values
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)

    df.columns = nombres_columnas  # Devuelvo los nombres de las columnas
    
    # Devolvemos el data frame ya normalizado
    return df
#####-----------------------------------------------------------------------------------------------######


#####---------------------FIN FUNCIONES-------------------------------------------------------------######


params_grid_search_petroleo = [{'sigma':1,'learning_rate':0.01},
          {'sigma':2,'learning_rate':0.01},
          {'sigma':3,'learning_rate':0.01}, 
          {'sigma':4,'learning_rate':0.01}, 
          {'sigma':5,'learning_rate':0.01}, 
          {'sigma':6,'learning_rate':0.01},
          {'sigma':1,'learning_rate':0.05},
          {'sigma':2,'learning_rate':0.05},
          {'sigma':3,'learning_rate':0.05}, 
          {'sigma':4,'learning_rate':0.05}, 
          {'sigma':5,'learning_rate':0.05}, 
          {'sigma':6,'learning_rate':0.05},                     
          {'sigma':1,'learning_rate':0.1},
          {'sigma':2,'learning_rate':0.1},
          {'sigma':3,'learning_rate':0.1}, 
          {'sigma':4,'learning_rate':0.1}, 
          {'sigma':5,'learning_rate':0.1}, 
          {'sigma':6,'learning_rate':0.1},                     
          {'sigma':1,'learning_rate':0.2},
          {'sigma':2,'learning_rate':0.2},
          {'sigma':3,'learning_rate':0.2}, 
          {'sigma':4,'learning_rate':0.2}, 
          {'sigma':5,'learning_rate':0.2}, 
          {'sigma':6,'learning_rate':0.2},
          {'sigma':1,'learning_rate':0.3},
          {'sigma':2,'learning_rate':0.3},
          {'sigma':3,'learning_rate':0.3}, 
          {'sigma':4,'learning_rate':0.3}, 
          {'sigma':5,'learning_rate':0.3}, 
          {'sigma':6,'learning_rate':0.3},
          {'sigma':1,'learning_rate':0.4},
          {'sigma':2,'learning_rate':0.4},
          {'sigma':3,'learning_rate':0.4}, 
          {'sigma':4,'learning_rate':0.4}, 
          {'sigma':5,'learning_rate':0.4}, 
          {'sigma':6,'learning_rate':0.4},
          {'sigma':1,'learning_rate':0.5},
          {'sigma':2,'learning_rate':0.5},
          {'sigma':3,'learning_rate':0.5}, 
          {'sigma':4,'learning_rate':0.5}, 
          {'sigma':5,'learning_rate':0.5}, 
          {'sigma':6,'learning_rate':0.5},
          {'sigma':1,'learning_rate':0.6},
          {'sigma':2,'learning_rate':0.6},
          {'sigma':3,'learning_rate':0.6}, 
          {'sigma':4,'learning_rate':0.6}, 
          {'sigma':5,'learning_rate':0.6}, 
          {'sigma':6,'learning_rate':0.6},
          {'sigma':1,'learning_rate':0.7},
          {'sigma':2,'learning_rate':0.7},
          {'sigma':3,'learning_rate':0.7}, 
          {'sigma':4,'learning_rate':0.7}, 
          {'sigma':5,'learning_rate':0.7}, 
          {'sigma':6,'learning_rate':0.7},
          {'sigma':1,'learning_rate':0.8},
          {'sigma':2,'learning_rate':0.8},
          {'sigma':3,'learning_rate':0.8}, 
          {'sigma':4,'learning_rate':0.8}, 
          {'sigma':5,'learning_rate':0.8}, 
          {'sigma':6,'learning_rate':0.8}]


