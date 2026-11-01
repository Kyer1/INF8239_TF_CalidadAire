# Ética, uso responsable y Green AI

## Ética y uso responsable
- **Datos:** públicos, sin información personal (mediciones ambientales), licencia CC BY 4.0 con cita a la fuente.
- **Riesgo principal:** un falso negativo puede dar una falsa sensación de seguridad a personas sensibles (asma, enfermedades cardíacas). Por eso se prioriza el recall de la clase *alto* junto con el F1 macro, y el tablero muestra la probabilidad y advierte que no sustituye a los avisos oficiales.
- **Representatividad:** una sola estación de Beijing entre 2010 y 2014; no debe extrapolarse a otras ciudades, como Santo Domingo, sin datos locales.
- **Transparencia:** código, partición, umbral, métricas y errores están publicados en el repositorio.

## Green AI
- Se midió el tiempo de ajuste (mediana de 3 repeticiones), el tiempo de inferencia y el tamaño del modelo serializado.
- Gradient boosting tarda ≈ 19 veces más en entrenar y ocupa ≈ 190 veces más que la regresión logística, a cambio de +0.025 de F1 macro y +0.044 de recall *alto* en prueba.
- En valores absolutos el costo es bajo (segundos en CPU y ≈ 1 MB), por lo que la mejora se considera justificada. Si el sistema tuviera que ejecutarse en un dispositivo muy limitado, la regresión logística sería la alternativa eficiente.
- Los tiempos dependen del equipo y no equivalen a consumo energético.
