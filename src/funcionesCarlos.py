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
import numpy as np
import pandas as pd
from imblearn.under_sampling import *
from imblearn.over_sampling import *
from imblearn.pipeline import *

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


"""
Función para combinar la técnica de over sampling SMOTE junto con las siguientes técnicas de under sampling:
◦ Tomek Links
◦ Edited Nearest Neighbors
◦ Condensed Nearest Neighbors
◦ Neighbourhood Cleaning Rule
◦ One Side Selection

La función devuelve un array con los datos remuestreados, siguiendo el orden definido en el comentario de arriba
"""
def overSampling_combinaciones(X,y,overSampling):
    
    over=0
    
    ### Tengo que hacer esta basurilla en vez de un switch-case porque en python 3.9 el match-case que vendría a ser
    ### el switch de python, no está aún implementado, se implementó en python 3.10, que todavía es demasiado reciente
    ### y va a dar problemas por no ser compatible con otras cosas
    if(overSampling == "SMOTE"):
        over = SMOTE()
    if(overSampling == "ADASYN"):
        over = ADASYN()
    if(overSampling == "BorderlineSMOTE"):
        over = BorderlineSMOTE()
    if(overSampling == "SVMSMOTE"):
        over = SVMSMOTE()
    if(overSampling == "KMeansSMOTE"):
        over = KMeansSMOTE(cluster_balance_threshold=0.05)
    
    if(over==0):
        print("Tecnica de oversampling incorrecta, revise el nombre...")
        return 0,0
    
    #----------Tomek Links---------------------
    # Hacemos el pipeline, indicando que métodos de under y over sampling usaremos
    under = TomekLinks()
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)


    # Generamos los datos sintéticos y los guardamos en las nuevas variables
    X_TomekLinks, y_TomekLinks = pipeline.fit_resample(X,y)
    
    
    #----------Edited Nearest Neighbors---------------------
    under = EditedNearestNeighbours()
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    X_EditedNearestNeighbours, y_EditedNearestNeighbours = pipeline.fit_resample(X,y)
    
    
    #----------Condensed Nearest Neighbour---------------------
    under = CondensedNearestNeighbour()
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    X_CondensedNearestNeighbour, y_CondensedNearestNeighbour = pipeline.fit_resample(X,y)    
    
    
    #----------Neighbourhood Cleaning Rule---------------------
    under = NeighbourhoodCleaningRule()
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    X_NeighbourhoodCleaningRule, y_NeighbourhoodCleaningRule = pipeline.fit_resample(X,y) 
    
    
    #----------One Side Selection---------------------
    under = OneSidedSelection()
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    X_OneSidedSelection, y_OneSidedSelection = pipeline.fit_resample(X,y)     
    
        
    X_array = np.array([X_TomekLinks,X_EditedNearestNeighbours,X_CondensedNearestNeighbour,X_NeighbourhoodCleaningRule,X_OneSidedSelection])
    y_array = np.array([y_TomekLinks,y_EditedNearestNeighbours,y_CondensedNearestNeighbour,y_NeighbourhoodCleaningRule,y_OneSidedSelection])
    
    
    return X_array,y_array



"""
Función para comprobar el error topológico y de cuantificación de la técnica de oversampling utilizada junto con las siguientes técnicas de under sampling:
◦ Tomek Links
◦ Edited Nearest Neighbors
◦ Condensed Nearest Neighbors
◦ Neighbourhood Cleaning Rule
◦ One Side Selection

La función evalua e imprime por pantalla el error topológico y de cuantificación de los datos contra el SOM.
"""
def SOM_errores(som,X_array,overSamplingUsado):

    US_tecnica = ["Tomek Links","Edited Nearest Neighbors","Condensed Nearest Neighbors","Neighbourhood Cleaning Rule","One Side Selection"]


    for cont,X in enumerate(X_array):      
        print("\n\nERROR de "+overSamplingUsado+" y "+US_tecnica[cont]+":")
        print("Topological error:",som.topographic_error(X))
        print("Quantization error:",som.quantization_error(X))

              
    return


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


