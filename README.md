# Laboratorio de transformación de intensidades

Aplicación en Streamlit para explorar funciones que aproximan la CDF de una imagen en escala de grises y transformar esa función mediante cuatro parámetros.

## Funciones incluidas

- Carga de imágenes comunes y conversión a escala de grises de 8 bits.
- Visualización del histograma y la CDF original.
- Aproximación de la CDF con interpolación PCHIP, interpolación lineal o una serie trigonométrica de Fourier.
- Control del número de puntos de control y, para Fourier, del número de términos.
- Transformación parametrizada:

  `G(r) = C * F_aprox(A*r + B) + D`

  `T(r) = r + 255 * (G(r) - F_aprox(r))`

- Comparación con la ecualización clásica (HE) y CLAHE.
- Métricas descriptivas, gráficas comparativas y descargas de la imagen transformada y las métricas en CSV.

Con `A=1`, `B=0`, `C=1` y `D=0`, la función transformada coincide con la aproximación original y el mapa resultante es la identidad `T(r)=r`. En consecuencia, la imagen debe permanecer sin cambios.

## Ejecutar en tu computador (opcional)

Se necesita Python instalado.

1. Abre una terminal dentro de esta carpeta.
2. Crea y activa un entorno virtual (recomendado).
3. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

4. Inicia la aplicación:

   ```bash
   streamlit run streamlit_app.py
   ```

Streamlit te mostrará una dirección local que puedes abrir en el navegador.

## Subir el proyecto a GitHub sin usar comandos

**Importante:** descomprime primero el ZIP en tu computador. Sube los archivos que contiene, no el ZIP como único archivo del repositorio. Streamlit necesita encontrar `streamlit_app.py` y `requirements.txt` en la raíz del repositorio.

1. Entra a [GitHub](https://github.com/) e inicia sesión.
2. Pulsa **New repository** (o entra en el menú `+` y elige **New repository**).
3. Ponle un nombre, por ejemplo `laboratorio-transformacion-intensidades`.
4. Elige **Public** si quieres que el código sea visible para cualquiera. Puedes usar **Private** si prefieres limitar quién ve el código, sujeto a la configuración de acceso que autorices al desplegar.
5. Pulsa **Create repository**.
6. En el repositorio nuevo, elige **Add file → Upload files**.
7. Arrastra los archivos y carpetas del proyecto descomprimido. Debes ver al nivel principal `streamlit_app.py`, `requirements.txt`, `README.md` y `.gitignore`.
8. Escribe un mensaje como `Primera versión de la aplicación` y pulsa **Commit changes**.

## Publicar la página en Streamlit Community Cloud

1. Entra a [Streamlit Community Cloud](https://share.streamlit.io/).
2. Inicia sesión con GitHub y autoriza el acceso solicitado.
3. Pulsa **Create app**.
4. Selecciona el repositorio recién creado, la rama `main` y el archivo principal `streamlit_app.py`.
5. Pulsa **Deploy**.
6. Cuando termine el despliegue, Streamlit te dará una dirección web terminada en `streamlit.app`.

La primera instalación puede tardar varios minutos. Cuando subas cambios al mismo repositorio, Streamlit normalmente actualizará la aplicación automáticamente.

## Cómo usar la aplicación

1. Carga una imagen desde el panel lateral.
2. Selecciona el método de aproximación y el número de puntos.
3. Mueve un parámetro a la vez para estudiar su efecto. Los valores neutros son `A=1`, `B=0`, `C=1`, `D=0`.
4. Revisa las pestañas de imagen/distribución, aproximación, transformación y comparación final.
5. Descarga la imagen transformada o la tabla de métricas si lo necesitas.

## Precauciones de interpretación

- La aplicación transforma la **CDF aproximada** mediante la diferencia `G(r)-F_aprox(r)` y suma esa diferencia a la intensidad original. No aplica directamente la CDF como mapa de intensidades.
- La aproximación de Fourier puede oscilar o superar el intervalo `[0, 1]`. Es un comportamiento posible de la aproximación, no necesariamente un error de visualización.
- La salida final se redondea y limita al rango `[0, 255]`. Por eso, para algunos parámetros diferentes de los neutros, varios valores de entrada pueden terminar en la misma intensidad de salida.
- No se garantiza que todos los parámetros produzcan una transformación creciente ni que mejoren todas las imágenes. Este laboratorio está diseñado para experimentar y analizar los resultados.
- El procesamiento de la página publicada se ejecuta en el servicio de Streamlit. No cargues imágenes personales, médicas, confidenciales o sensibles si no tienes autorización para procesarlas en un servicio externo.

## Archivos del proyecto

- `streamlit_app.py`: aplicación completa.
- `requirements.txt`: dependencias de Python.
- `.gitignore`: archivos locales que no conviene subir.
- `README.md`: instrucciones de instalación y publicación.
