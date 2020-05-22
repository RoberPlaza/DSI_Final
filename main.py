import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os import listdir
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

force_rw    = True
ccaa_data   = []
data_path   = "data/splitted/"


### Partimos los datos si no están ya partidos
if not Path( "data/splitted/" ).exists() or force_rw:
    Path( "data/splitted/" ).mkdir( parents=True, exist_ok=True )
    data    = pd.read_csv( "data/covid_data.csv" )
    
    for ca in data.CCAA.unique():
        df  = data.loc[ data[ 'CCAA' ] == ca ]
        del df[ 'CCAA' ] # Borramos el nombre de la comunidad
        df.to_csv( f"data/splitted/{ ca }.csv", index=False )

### Cargamos los datos de las diferentes comunidades autónomas en diferentes dataframes
ccaa_data   = [ pd.read_csv( f"{ data_path }{ file }" ) for file in sorted( listdir( data_path ) ) ]


