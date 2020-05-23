#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dtw import dtw

### Definimos distancias como lambdas para el DTW
manhattan_distance      = lambda x, y :np.abs( x - y )          # Distancia manhattan para escalares
euclidean_distance      = lambda x, y :np.sqrt( x * x - y * y ) # Distancia euclídea para escalares
euclidean_distance_v    = lambda x, y :np.linalg.norm( x - y )  # Distancia euclídea para vectores


### Algunas constantes globales
target_ccaa = "CM"                                  # Nos centramos en Castilla-La Mancha
base_data   = pd.read_csv( "data/covid_data.csv" )  # Dataframe con todo el contenido de la evolución del COVID-19                   
ccaa_names  = base_data[ 'CCAA' ].unique()          # Nombres de las comunidades autónomas
num_cols    = [ 'CASOS', 'Hospitalizados', 'UCI', 'Fallecidos', 'Recuperados' ] # Columnas con valores numéricos


def generate_dtw_plots():
    clm_data    = pd.DataFrame( base_data.loc[ base_data[ 'CCAA' ] == target_ccaa ].reset_index()[ num_cols ] )
    ccaa_data   = { ca : pd.DataFrame( base_data.loc[ base_data[ 'CCAA' ] == ca ].reset_index()[ num_cols ] ) 
                  for ca in ccaa_names if ca != target_ccaa }        
    
    for ca in ccaa_names:
        if ca != target_ccaa:
            distance, cost_matrix, acc_cost_matrix, path = dtw( clm_data.to_numpy(), ccaa_data[ ca ].to_numpy(), dist = euclidean_distance_v )
            plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
            plt.plot(path[0], path[1], 'w')
            plt.xlabel( "Datos de CM" )
            plt.ylabel( f"Datos de { ca }" )
            plt.savefig( f"images/DTW_{ ca }.pdf" )
            plt.clf()


### Guardamos la serie temporal de Castilla-La Mancha.
clm_data    = pd.DataFrame( base_data.loc[ base_data[ 'CCAA' ] == target_ccaa ].reset_index()
              [ [ 'FECHA', 'CASOS', 'Hospitalizados', 'UCI', 'Fallecidos', 'Recuperados' ] ] )

### Guardamos el resto de series temporales del covid.
ccaa_data   = { ca : pd.DataFrame( base_data.loc[ base_data[ 'CCAA' ] == ca ].reset_index()
              [ [ 'FECHA', 'CASOS', 'Hospitalizados', 'UCI', 'Fallecidos', 'Recuperados' ] ] ) 
              for ca in ccaa_names if ca != target_ccaa }

### Calculamos las distancias de cada comunidad autónoma y asset con Castilla-La Mancha
### La distancia, en lugar de ser el peso es la longitud del camino del algoritmo DTW
distancias  = { column : { ca : len( dtw( clm_data[ column ].to_numpy(), ccaa_data[ ca ][ column ], dist = manhattan_distance )[ 3 ][ 0 ] )
              for ca in ccaa_names if ca != target_ccaa } for column in num_cols }
