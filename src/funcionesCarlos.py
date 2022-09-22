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
    
    # Damos un valor por si no le asignamos ninguna en los ifs
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
        return 0,0,0
    
#--------------------------------------------------------------------------------------------------------    
    ## Vamos a realizar el método de oversampling seleccionado, y quedarnos también con ese conjunto de datos sintéticos
    ## generados, para poder evaluarlos aislados
     
    # Genero el nuevo conjunto
    X_over, y_over = over.fit_resample(X,y)
    
    # Miro cuantos elementos tiene
    num_elems = X_over.shape[0]
    
    # Miro cuantos son nuevos comparandolo con el número de elementos originales
    num_elems_nuevos = num_elems - X.shape[0]
    
    # Introduzco solo los nuevos en el array de nuevos elementos
    over_array_elems_nuevos = X_over[-num_elems_nuevos:]
    # Introduzco solo las nuevas etiquetas
    over_array_etiquetas_nuevas = y_over[-num_elems_nuevos:]
#--------------------------------------------------------------------------------------------------------     
    
    #Ahora vamos a aplicar distintos undersampling a nuestro conjunto que ya ha sido "oversampleado"
    
    #----------Tomek Links---------------------
    under = TomekLinks()
    
    X_TomekLinks, y_TomekLinks = under.fit_resample(X_over,y_over)
    
       
    #----------Edited Nearest Neighbors---------------------
    under = EditedNearestNeighbours()
    
    X_EditedNearestNeighbours, y_EditedNearestNeighbours = under.fit_resample(X_over,y_over)
    
    
    #----------Condensed Nearest Neighbour---------------------
    under = CondensedNearestNeighbour()
    
    X_CondensedNearestNeighbour, y_CondensedNearestNeighbour =  under.fit_resample(X_over,y_over)   
    
    
    #----------Neighbourhood Cleaning Rule---------------------
    under = NeighbourhoodCleaningRule()
    
    X_NeighbourhoodCleaningRule, y_NeighbourhoodCleaningRule = under.fit_resample(X_over,y_over)
    
    
    #----------One Side Selection---------------------
    under = OneSidedSelection()
    
    X_OneSidedSelection, y_OneSidedSelection = under.fit_resample(X_over,y_over)   
    
        
    X_array = np.array([X_TomekLinks,X_EditedNearestNeighbours,X_CondensedNearestNeighbour,X_NeighbourhoodCleaningRule,X_OneSidedSelection])
    y_array = np.array([y_TomekLinks,y_EditedNearestNeighbours,y_CondensedNearestNeighbour,y_NeighbourhoodCleaningRule,y_OneSidedSelection])

    
    return X_array,y_array,over_array_elems_nuevos,over_array_etiquetas_nuevas



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


