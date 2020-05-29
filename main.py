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
# distance_f  = lambda pair :pair[ 1 ].distance

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
clm_x       = clm_data.iloc[ : train_days,  : ][ target_cols ].to_numpy()

### Creamos los conjuntos de train y test
for i in range( train_ccaa ):
    data_as_np      = ccaa_data[ similar_ca[ i ] ][ target_cols ].to_numpy()
    train_x[ i ]    = data_as_np[ : train_days ]
    train_y[ i ]    = data_as_np[ lookback : train_days + lookback ]

### Cambiamos la forma de los datos para que la red LSTM los admita ( número_muestras, número_columnas, número_filas )
train_x     = np.reshape( train_x, ( train_x.shape[ 0 ], train_x.shape[ 2 ], train_x.shape[ 1 ] ) )
train_y     = np.reshape( train_y, ( train_y.shape[ 0 ], train_y.shape[ 2 ], train_y.shape[ 1 ] ) )
clm_x       = np.reshape( clm_x, ( 1, 1, clm_x.shape[ 0 ] ) )

### Creamos el modelo de predicción con keras
model       = Sequential( )
model.add( LSTM( train_days, return_sequences = True, input_shape = ( len( target_cols ), train_days ) ) )

model.add( LSTM( train_days * 21, return_sequences = True ) )
model.add( Dense( 30 ) )

### Entrenamos el modelo
model.compile( loss='mean_squared_error', optimizer='adam' )
model.fit( train_x, train_y, epochs = 20000, batch_size = train_ccaa, verbose = 1 )

### Guardamos el modelo en formato hdf5
model.save( "model/covid_predictor.h5" )

### Predecimos los próximos 30 días de castilla-La Mancha
prediction = model.predict( clm_x, batch_size = len( target_cols ), verbose = 2 )

### Y cambiamos el formato para que sea un vector de dimensión única
prediction = np.resize( prediction.astype( int ), ( 30 ) )

### Generamos unos gráficos para validación visual
plt.plot( prediction, label = "Predicción" )
plt.plot( clm_data.iloc[ 30 : 60, : ][ target_cols ].to_numpy(), label = "Datos experimentales" )
plt.title( "Predicción de Castilla-La Mancha" )
plt.legend()
plt.savefig( "images/Prediction_CLM.pdf" )
plt.show()
plt.clf()

for ccaa in similar_ca:
    exp_data    = ccaa_data[ ccaa ][ target_cols ].to_numpy()[ 30 : 60, : ]
    exp_data    = np.resize( exp_data, ( 1, 1, 30 ) )
    
    pred        = model.predict( exp_data, batch_size = 1 )
    pred        = np.resize( pred.astype( int ), ( 30 ) )
    
    exp_data    = np.resize( exp_data, ( 30 ) )
    
    plt.plot( pred, label = "Predicción" )
    plt.plot( exp_data, label = "Datos experimentales" )
    plt.title( f"Predicción y datos empíricos de { ccaa }" )
    plt.legend()
    plt.savefig( f"images/Prediction_{ ccaa }.pdf" )
    plt.show()
    plt.clf()
