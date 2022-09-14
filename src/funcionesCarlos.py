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