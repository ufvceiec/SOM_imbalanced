#####-------------------------------------------------------######
"""
Archivo con funciones hechas por Carlos
"""
#####-------------------------------------------------------######
#####-------------------------------------------------------######


#### LIBRERIAS 

from sklearn import preprocessing
import pandas as pd

#### FIN LIBRERIAS



#### FUNCIONES

"""
   Normalización mediante la técnica minmax
"""
def min_max(df):
    
    #Usamos el MinMax que viene con SKLearn ---> from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    x=df.values
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    
    return df