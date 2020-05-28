#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dtw import dtw

from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential


### Función distancia. La defino aquí por si la quiero cambiar más adelante.
distance_f  = lambda pair :len( pair[ 1 ].stepsTaken )

### Algunas constantes globales
target_ccaa = "CM"                                                              # Nos centramos en Castilla-La Mancha
target_col  = "CASOS"                                                           # Columna a tratar
train_ccaa  = 6                                                                 # Número de comunidades autónomas las cuales vamos a entrenar
train_prcnt = 0.67                                                              # Proporción de entrenamiento/prueba
lookback    = 7                                                                 # Memoria de la red neuranal, 1 semana
base_data   = pd.read_csv( "data/covid_data.csv" )                              # Dataframe con todo el contenido de la evolución del COVID-19                   
ccaa_names  = base_data[ 'CCAA' ].unique()                                      # Nombres de las comunidades autónomas
num_cols    = [ 'CASOS', 'Hospitalizados', 'UCI', 'Fallecidos', 'Recuperados' ] # Columnas con valores numéricos


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

### Precalculamos algunos índices
train_size  = int( clm_data.shape[ 0 ] * train_prcnt )                          # Tamaño de datos de entrenamiento
test_size   = clm_data.shape[ 0 ] - train_size                                  # Tamaño de datos de test

### Preinicializamos los arrays de train y test
train_x     = np.ones( ( train_ccaa, train_size ) )                             # Datos de entreno X
train_y     = np.ones( ( train_ccaa, train_size ) )                             # Datos de entreno Y
test_x      = np.ones( ( train_ccaa, test_size ) )                              # Datos de prueba
clm_x       = clm_data.iloc[ : -test_size,  : ][ target_col ].to_numpy()

### Creamos los conjuntos de train y test
for i in range( train_ccaa ):
    data_as_np      = ccaa_data[ similar_ca[ i ] ][ target_col ].to_numpy()
    train_x[ i ]    = data_as_np[ : int( data_as_np.size * train_prcnt ) ]
    train_y[ i ]    = data_as_np[ lookback : int( data_as_np.size * train_prcnt ) + lookback ]
    test_x[ i ]     = data_as_np[ int( data_as_np.size * train_prcnt ) : ]

### Cosa estúpida que no entiendo porqué hay que hacer
train_x     = np.reshape( train_x, ( train_x.shape[ 0 ], 1, train_x.shape[ 1 ] ) )
train_y     = np.reshape( train_y, ( train_y.shape[ 0 ], 1, train_y.shape[ 1 ] ) )
clm_x       = np.reshape( clm_x, ( 1, 1, 60 ) )

model       = Sequential()
# =============================================================================
# model.add( LSTM( 30, return_sequences = True, input_shape = ( 1, train_size ) ) )
# model.add( Dropout( 0.5 ) )
# 
# model.add( LSTM( 30, return_sequences = True ) )
# model.add( Dropout( 0.5 ) )
# 
# model.add( LSTM( 30, return_sequences = True ) )
# model.add( Dropout( 0.5 ) )
# =============================================================================

model.add( LSTM( 60, return_sequences = True, batch_input_shape = ( train_ccaa, 1, 60 ) ) )
model.add( Dropout( 0.5 ) )

model.add( LSTM( 60, return_sequences = True ) )
model.add( Dropout( 0.5 ) )

model.add( LSTM( 60, return_sequences = True ) )
model.add( Dropout( 0.5 ) )

model.add( LSTM( 60, return_sequences = True ) )
model.add( Dropout( 0.5 ) )

model.add( LSTM( 60, return_sequences = True ) )
model.add( Dropout( 0.5 ) )

model.add( LSTM( 60, return_sequences = True ) )
model.add( Dropout( 0.5 ) )

model.add( Dense( 60 ) )

model.summary()

model.compile( loss='mean_squared_error', optimizer='adam' )
model.fit( train_y, train_x, epochs = 1000, batch_size = 6, verbose = 1 )
#model.predict_generator( clm_data, steps = 1 )
