# Dataset Card · Beijing PM2.5

## Identificación
- **Nombre:** Beijing PM2.5 (archivo `PRSA_data_2010.1.1-2014.12.31.csv`)
- **Fuente:** UCI Machine Learning Repository, id 381 — https://archive.ics.uci.edu/dataset/381
- **Descarga directa:** https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip
- **Autor:** Song Chen. Trabajo asociado: Liang, X. et al. (2015), *Proceedings of the Royal Society A*.
- **Cita:** Chen, S. (2015). Beijing PM2.5 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5JS49
- **Licencia:** CC BY 4.0. **Periodo:** 1 de enero de 2010 a 31 de diciembre de 2014. **Frecuencia:** horaria.
- **Versión:** identificada por el SHA-256 que imprime `scripts/download_data.py`.

## Diccionario de datos

| Columna | Tipo | Descripción | Unidad / valores |
|---|---|---|---|
| No | entero | número de fila | — |
| year, month, day, hour | entero | fecha y hora de la medición | — |
| pm2.5 | numérico | concentración de PM2.5 (Embajada de EE. UU.) | µg/m³; 2,067 ausentes |
| DEWP | numérico | punto de rocío | °C |
| TEMP | numérico | temperatura | °C |
| PRES | numérico | presión atmosférica | hPa |
| cbwd | categórico | dirección del viento combinada | NW, NE, SE, cv (calma/variable) |
| Iws | numérico | velocidad del viento acumulada | m/s |
| Is | numérico | horas de nieve acumuladas | h |
| Ir | numérico | horas de lluvia acumuladas | h |

## Calidad observada
- 43,824 horas continuas, sin horas faltantes ni marcas de tiempo duplicadas.
- Solo `pm2.5` tiene ausentes (4.7 %), sobre todo en 2010–2012.
- PM2.5: mediana 72 µg/m³, máximo 994; el 48.5 % de las horas supera 75 µg/m³.

## Población cubierta y excluida
Una sola estación (Embajada de EE. UU., Beijing) y una estación meteorológica (aeropuerto). No representa otras zonas de la ciudad, otras ciudades ni años posteriores a 2014, cuando cambiaron las políticas de control de emisiones.

## Riesgos y usos prohibidos
No usar para emitir alertas oficiales de salud ni para decisiones sobre personas. Los datos no incluyen otros contaminantes (PM10, O₃, NO₂) ni pronósticos meteorológicos.
