#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dtw import dtw

from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential


### Función distancia. La defino aquí por si la quiero cambiar más adelante.
distance_f  = lambda pair :len( pair[ 1 ].stepsTaken )

### Algunas constantes globales
target_ccaa = "CM"                                                              # Nos centramos en Castilla-La Mancha
target_col  = "CASOS"                                                           # Columna a tratar
train_ccaa  = 7                                                                 # Número de comunidades autónomas las cuales vamos a entrenar
train_prcnt = 0.67                                                              # Proporción de entrenamiento/prueba
lookback    = 3                                                                 # Memoria del LLSTM
base_data   = pd.read_csv( "data/covid_data.csv" )                              # Dataframe con todo el contenido de la evolución del COVID-19                   
ccaa_names  = base_data[ 'CCAA' ].unique()                                      # Nombres de las comunidades autónomas
num_cols    = [ 'CASOS', 'Hospitalizados', 'UCI', 'Fallecidos', 'Recuperados' ] # Columnas con valores numéricos
train_data_x= pd.DataFrame()                                                    # Datos de entreno X
train_data_y= pd.DataFrame()                                                    # Datos de entreno Y
test_data   = pd.DataFrame()                                                    # Datos de prueba


### Guardamos la serie temporal de Castilla-La Mancha.
clm_data    = pd.DataFrame( base_data.loc[ base_data[ 'CCAA' ] == target_ccaa ].reset_index()
              [ [ 'FECHA', 'CASOS', 'Hospitalizados', 'UCI', 'Fallecidos', 'Recuperados' ] ] )

### Guardamos el resto de series temporales del covid.
ccaa_data   = { ca : pd.DataFrame( base_data.loc[ base_data[ 'CCAA' ] == ca ].reset_index()
              [ [ 'FECHA', 'CASOS', 'Hospitalizados', 'UCI', 'Fallecidos', 'Recuperados' ] ] ) 
              for ca in ccaa_names if ca != target_ccaa }

### Calculamos las distancias de cada comunidad autónoma y asset con Castilla-La Mancha
### La distancia, en lugar de ser el peso es la longitud del camino del algoritmo DTW
distancias  = { ca : dtw( clm_data[ target_col ].to_numpy(), ccaa_data[ ca ][ target_col ] )
              for ca in ccaa_names if ca != target_ccaa }

### Guardamos los nombres de las comunidades que más separecen a CLM en una lista
similar_ca  = [ pair[ 0 ] for pair in sorted( distancias.items(), key = distance_f )[ :train_ccaa ] ]

### Creamos los conjuntos de train y test
for train_ca in similar_ca:
    data_as_np                  = ccaa_data[ train_ca ][ target_col ].to_numpy()
    train_data_x[ train_ca ]    = data_as_np[ : int( data_as_np.size * train_prcnt ) ]
    train_data_y[ train_ca ]    = data_as_np[ : int( data_as_np.size * train_prcnt ) + lookback ]
    test_data[ train_ca ]       = data_as_np[ int( data_as_np.size * train_prcnt ) : ]

model       = Sequential()
model.add( LSTM( 5 ) )
model.add( Dense( 1 ) )
model.compile( loss='mean_squared_error', optimizer='adam' )
model.fit( train_data_x.to_numpy(), train_data_y.to_numpy() )