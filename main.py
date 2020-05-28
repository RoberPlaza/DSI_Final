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
target_cols = [ "CASOS" ]                                                       # Columna a tratar
train_ccaa  = 5                                                                 # Número de comunidades autónomas las cuales vamos a entrenar
train_days  = 30                                                                # Número de días de entrenamiento para el dataset
lookback    = 30                                                                # Memoria de la red neuranal
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
              for target_col in target_cols for ca in ccaa_names if ca != target_ccaa }

### Guardamos los nombres de las comunidades que más separecen a CLM en una lista
similar_ca  = [ pair[ 0 ] for pair in sorted( distancias.items(), key = distance_f )[ :train_ccaa ] ]

### Preinicializamos los arrays de train y test
train_x     = np.ones( ( train_ccaa, train_days, len( target_cols ) ) )         # Datos de entreno X
train_y     = np.ones( ( train_ccaa, train_days, len( target_cols ) ) )         # Datos de entreno Y
clm_x       = clm_data.iloc[ : -train_days * 2 ,  : ][ target_cols ].to_numpy()

### Creamos los conjuntos de train y test
for i in range( train_ccaa ):
    data_as_np      = ccaa_data[ similar_ca[ i ] ][ target_cols ].to_numpy()
    train_x[ i ]    = data_as_np[ : train_days ]
    train_y[ i ]    = data_as_np[ lookback : train_days + lookback ]

### Cosa estúpida que no entiendo porqué hay que hacer
train_x     = np.reshape( train_x, ( train_x.shape[ 0 ], train_x.shape[ 2 ], train_x.shape[ 1 ] ) )
train_y     = np.reshape( train_y, ( train_y.shape[ 0 ], train_y.shape[ 2 ], train_y.shape[ 1 ] ) )
clm_x       = np.reshape( clm_x, ( 1, 1, clm_x.shape[ 0 ] ) )

model       = Sequential()
model.add( LSTM( train_days, return_sequences = True, batch_input_shape = ( train_ccaa, 1, train_days ) ) )

model.add( LSTM( train_days * 2, return_sequences = True ) )
model.add( Dropout( 0.20 ) )

model.add( LSTM( train_days * 7, return_sequences = True ) )
model.add( Dropout( 0.20 ) )

model.add( LSTM( train_days * 30, return_sequences = True ) )
model.add( Dropout( 0.20 ) )

model.add( LSTM( train_days * 30, return_sequences = True ) )
model.add( Dropout( 0.20 ) )

model.add( LSTM( train_days * 2, return_sequences = True ) )
model.add( Dropout( 0.20 ) )

model.add( LSTM( train_days * 7, return_sequences = True ) )
model.add( Dense( 30 ) )

model.summary()

model.compile( loss='mean_squared_error', optimizer='adam' )
model.fit( train_y, train_x, epochs = 2500, batch_size = train_ccaa, verbose = 1 )
model.predict_generator( clm_x, steps = 1 )
