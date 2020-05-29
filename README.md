# DSI Final
Este es el repositorio del trabajo final de la asignatura «Desarrollo de sistemas inteligentes», impartida en la Escuela Superior de Informática de Ciudad Real perteneciente a la Universidad de Castilla-La Mancha. El grupo responsable de este proyecto consta únicamente del alumno Roberto Plaza Romero (Roberto.Plaza {at} alu {dot} uclm {dot} es).

## Explicación
El fin de este proyecto es el de predecir la evolución de casos de COVID-19 en Castilla-La Mancha, para eso se ha creado un modelo de red neuronal LSTM y se ha entrenado con la evolución de casos de las cinco comunidades autónomas más parecidas a Castilla-La Mancha. Esta medida de similitud se calcula usando el algoritmo de Dynamic Time Warping. De forma común el DTW calcula la distancia en valor absoluto, no obstante, la función distancia varía, en vez de el valor completo se usará la cantidad de pasos que el algoritmo de hasta estar completado.

## How-To
Este modelo se ha desarrollado usando Keras en un entorno anaconda. Hará falta instalar los siguientes paquetes conda: dtw (no confundir con R Dtw) y tensorflow, para mi caso he usado la distribución de gpu en lugar de la de cpu, pero la funcionalidad debería ser la misma. Pueden ser instalados con los comandos:

```console
conda install -c freemapa dtw
conda install -c anaconda tensorflow-gpu
```

## Resultados
El modelo final puede descargarse en formato h5 como en la sección de "releases" dentro de este repositorio. Este formato puede cargarse directamente utilizando keras o cualquier otra librería que soporte el formato.

En cuanto a la validación con datos ya conocidos, el modelo presenta un sesgo a la baja en cuanto al número de infectados, pero en general, la función es parecida y puede servir como guía. Otros modelos eran capaces de preveer los casos anormales mejor, no obstante, no ofrecían una visión genérica de la evolución de infectados mejor que la del modelo final.