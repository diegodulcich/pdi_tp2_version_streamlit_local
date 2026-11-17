# Pipeline de restauración fotográfica con difusión

Este documento describe el flujo que implementa actualmente el MVP. Está escrito
como guía educativa: sigue los datos desde la carga de una fotografía hasta la
descarga del resultado y explica qué hace cada módulo.

> **Idea central:** el sistema no recupera los píxeles originales que faltan. La
> difusión propone una reconstrucción plausible para la región que marca la
> persona usuaria.

## 1. Vista general

```text
Persona usuaria
    |
    | fotografía JPG/PNG o ejemplo local
    v
app.py: carga y preparación
    |-- orienta usando metadatos EXIF
    |-- convierte a escala de grises
    |-- limita el lado mayor a 512 px
    v
app.py + streamlit-drawable-canvas: máscara manual
    |-- la persona pinta en rojo sobre el daño
    |-- rojo se convierte en blanco (reconstruir)
    |-- resto se convierte en negro (conservar)
    v
Selección de modo en app.py
    |-- demo local: filtro mediano, sin modelos
    |-- pipeline de IA: Stable Diffusion 1.5 Inpainting en CPU
    |   |-- instrucción + fotografía + máscara + parámetros
    |   |-- genera una propuesta dentro de la máscara
    |-- ambos modos conservan el exterior original de la máscara
    v
restaurador.imagenes.crear_mapa_diferencias()
    |-- calcula cambios de intensidad entre entrada y salida
    v
restaurador.procesamiento.analizar_imagen()
    |-- solo en el pipeline completo: RT-DETR y BLIP
    |-- ambas imágenes se analizan para comparar
    v
app.py: presentación y descarga
    |-- antes/después, máscara y mapa; análisis solo con IA completa
    |-- exportación del resultado como PNG
```

## 2. Inicio de la aplicación

La aplicación se inicia desde la carpeta del proyecto con:

```powershell
uv run --python 3.12 streamlit run app.py --server.address 127.0.0.1
```

También se puede usar `ejecutar_app.bat`. `uv` prepara o reutiliza el entorno
virtual definido por `pyproject.toml` y `uv.lock`; Streamlit ejecuta `app.py` y
sirve la interfaz solo en la máquina local. La ventana de terminal debe seguir
abierta mientras se use la aplicación.

Al comenzar una sesión, Python importa los módulos y Streamlit ejecuta el script
para construir la interfaz. Streamlit vuelve a ejecutar el script cuando cambian
los controles, pero el procesamiento solo se ejecuta al presionar el botón del
modo elegido. Si no hay ejemplos locales disponibles, la app solicita cargar una
foto propia.

## 3. Entrada y preparación de la fotografía

### 3.1 Selección de la imagen

En `app.py`, `leer_imagen_seleccionada()` toma una de estas entradas:

- Un archivo propio en formato JPG, JPEG o PNG.
- Uno de los ejemplos incluidos en `modelos_difusion/datos/`.

Si no se puede abrir el archivo, la interfaz muestra un error. La imagen se lee
desde memoria; el programa no la guarda automáticamente como archivo original.

### 3.2 Orientación, escala de grises y resolución

`restaurador/imagenes.py:preparar_imagen()` aplica estas operaciones:

1. Corrige la orientación de la foto según los metadatos EXIF.
2. Convierte la imagen a escala de grises.
3. Reduce proporcionalmente la imagen cuando algún lado supera 512 píxeles.
4. Ajusta las dimensiones a múltiplos de 8, requeridos por el modelo de
   difusión.
5. Convierte la imagen gris a RGB de tres canales para que sea compatible con
   los procesadores de los modelos.

Aunque el modo resultante es RGB, sus tres canales contienen el mismo valor: la
fotografía sigue viéndose en blanco y negro. La imagen procesada, no la entrada
original a resolución completa, es la base para la máscara, el inpainting, el
análisis y la descarga.

**Por qué 512 px:** reduce el trabajo y la memoria requeridos en el hardware de
desarrollo. No agrega detalle que no estuviera en la fotografía.

## 4. Creación de la máscara

El lienzo de `streamlit-drawable-canvas` muestra la imagen preparada como fondo.
La persona dibuja con un pincel rojo sobre el rayón, la rotura u otra región que
quiera reconstruir. El pincel tiene un grosor inicial de 14 píxeles de la imagen
de trabajo, y el lienzo ofrece controles para deshacer, rehacer y limpiar.

`extraer_mascara_canvas()` convierte el resultado del lienzo en una máscara
binaria:

| Valor | Significado |
|---|---|
| Blanco (`255`) | Región que se puede reconstruir |
| Negro (`0`) | Región que se debe conservar |

La interfaz muestra una vista previa roja de la máscara. Antes de ejecutar la
difusión, `normalizar_mascara()` la ajusta al tamaño de la fotografía con
interpolación de vecino más cercano. Esto evita introducir valores grises en los
bordes de una máscara que debe ser binaria.

Si no se dibujó ningún trazo, la máscara está vacía y el procesamiento se detiene
con un mensaje que pide marcar la zona.

**La máscara representa una decisión de la persona:** si cubre demasiado, la IA
tiene más libertad para inventar; si cubre muy poco, parte del daño puede quedar
sin tratar.

## 5. Parámetros de generación

Antes de ejecutar, la interfaz solicita:

- **Instrucción (`prompt`):** texto que orienta la apariencia deseada de la zona
  generada. El valor inicial está en inglés. Es una condición para la generación,
  no una observación ni evidencia del contenido histórico.
- **Pasos de difusión:** número de iteraciones que ejecuta el proceso generativo.
  Más pasos suelen implicar más tiempo y pueden refinar la propuesta; no hacen
  que sea más verdadera. El mínimo y valor inicial es 5.
- **Fuerza:** controla cuánto se permite reconstruir la parte enmascarada frente
  a conservar la imagen de entrada. El valor inicial es 0,85; con fuerza alta la
  propuesta tiene más libertad.
- **Semilla:** inicializa el generador aleatorio. Usar la misma semilla y los
  mismos parámetros facilita repetir una prueba en el mismo entorno. No garantiza
  resultados idénticos entre distintas versiones de software o hardware.

`clave_solicitud()` calcula una huella de la imagen, la máscara y los parámetros.
La interfaz la usa para advertir si se cambió una entrada después de obtener un
resultado: no presenta silenciosamente un resultado anterior como si
correspondiera a los parámetros nuevos.

## 6. Inpainting con Stable Diffusion

La función `reparar_imagen()` en `restaurador/procesamiento.py` ejecuta la
reconstrucción:

1. Convierte imagen y máscara a los formatos esperados.
2. Comprueba que la máscara contenga una región marcada.
3. Obtiene la tubería de inpainting desde `cargar_inpainting()`.
4. Crea un generador aleatorio de CPU con la semilla elegida.
5. Envía al modelo la instrucción, la imagen, la máscara, la fuerza y el número
   de pasos.
6. Recibe una imagen generada.
7. Combina la salida con la imagen original para conservar el exterior de la
   máscara.

### Modelo y ejecución

El modelo seleccionado es
[`stable-diffusion-v1-5/stable-diffusion-inpainting`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting),
con una revisión concreta fijada en `restaurador/modelos.py`. Está especializado
en completar regiones enmascaradas.

`cargar_inpainting()` usa Diffusers, ejecuta el modelo en CPU con precisión
FP32 y activa el *attention slicing* para reducir memoria temporal. Es una ruta
de compatibilidad para la Intel UHD del equipo: esta versión no usa CUDA ni
OpenVINO. La primera ejecución necesita conexión a Internet para descargar los
pesos; Hugging Face los mantiene en su caché local para usos posteriores.

### Demostración local sin modelos

En equipos sin GPU o para validar el flujo sin descargar los modelos se puede
seleccionar **Demostración local (sin IA)** en `app.py`. Esta ruta aplica un
filtro mediano de 5 × 5 y sustituye únicamente los píxeles dentro de la máscara.
No carga pesos ni ejecuta detección o captioning. Puede disimular algunos daños
pequeños, pero no completa el contenido faltante; es una prueba funcional de la
interfaz y de la composición enmascarada.

La demostración usa la misma comparación y descarga que el pipeline completo.
Al cambiar de modo, la interfaz invalida el resultado anterior para no mostrar
un resultado generado por otro flujo como si correspondiera a la selección
actual.

### Composición final

`componer_solo_mascara()` construye el resultado de forma explícita:

- Dentro de la máscara, utiliza los píxeles generados y los vuelve a escala de
  grises.
- Fuera de la máscara, utiliza los píxeles originales de la imagen procesada.

Por eso la salida debe conservar exactamente las zonas no enmascaradas. Esta
composición limita cambios accidentales fuera del daño, pero no valida el
contenido nuevo dentro de la máscara.

## 7. Mapa de diferencias

`crear_mapa_diferencias()` compara la imagen de trabajo con la reconstrucción:

1. Calcula la diferencia absoluta de cada canal RGB.
2. Promedia las diferencias de los tres canales para cada píxel.
3. Colorea el mapa: tonos oscuros representan menor diferencia y tonos claros,
   mayor diferencia.
4. Cuenta el porcentaje de píxeles cuya diferencia media supera `10/255`.

El porcentaje es una medida visual simple de cuánto cambió la imagen procesada.
No es una métrica de calidad, de fidelidad ni de autenticidad. En este MVP, como
la composición mantiene intacto el exterior de la máscara, los cambios deberían
concentrarse en la región marcada.

## 8. Análisis visual antes y después

`ejecutar_pipeline()` analiza la imagen original procesada y la reconstruida.
Para cada versión, `analizar_imagen()` obtiene dos resultados:

### 8.1 Detección de objetos con RT-DETR

El modelo `PekingU/rtdetr_r50vd` busca categorías de objetos conocidas y devuelve
etiqueta, puntaje de confianza y caja delimitadora. La interfaz presenta
detecciones con puntaje igual o mayor a `0,50`.

Comparar las listas antes y después puede señalar diferencias generales en
objetos detectados. No demuestra que un objeto se haya preservado, ni que un
detalle reconstruido sea correcto. Un objeto puede no detectarse por el tipo de
fotografía, su tamaño, el daño o las limitaciones del detector.

### 8.2 Descripción con BLIP

`Salesforce/blip-image-captioning-base` genera una descripción breve de cada
imagen, con un máximo de 40 tokens nuevos. El modelo produce captions en inglés;
la aplicación los presenta tal como fueron generados y no los traduce.

Un caption resume contenido general, puede omitir detalles y no debe leerse como
una explicación fiable de lo que había dentro de la zona dañada.

### 8.3 Caché y consumo de memoria

`cargar_inpainting()`, `cargar_detector()` y `cargar_descriptor()` usan
`st.cache_resource`, por lo que Streamlit reutiliza cada modelo cargado durante
la sesión en lugar de volver a descargarlo o inicializarlo ante cada interacción.
Los modelos ejecutan sus inferencias en CPU. Aunque las llamadas de análisis se
hacen por imagen, los recursos cacheados pueden seguir ocupando RAM a la vez; el
código actual no los descarga de memoria entre etapas.

## 9. Presentación y exportación

`ResultadoPipeline` reúne las imágenes, la máscara, el mapa, el porcentaje de
cambio, las detecciones y las descripciones. `app.py` presenta:

1. Original y propuesta reconstruida lado a lado.
2. Mapa de diferencias y porcentaje de píxeles modificados sobre el umbral.
3. Caption y detecciones de la versión original.
4. Caption y detecciones de la versión reconstruida.
5. Una explicación de los límites de cada resultado.
6. Un botón para descargar la reconstrucción como PNG.

En la demostración local se omiten las detecciones y descripciones; la interfaz
lo indica explícitamente.

El PNG descargado tiene la resolución procesada (hasta 512 px de lado) y se
mantiene en blanco y negro. No se amplía de vuelta al tamaño de la fotografía
original.

## 10. Errores y límites del flujo

- Si el archivo no puede abrirse, la aplicación comunica que no pudo leerlo.
- Si no se dibujó una máscara, solicita marcar una región antes de generar.
- Si ocurre un error conocido durante el procesamiento, la interfaz muestra el
  mensaje y el detalle técnico de la excepción para facilitar el diagnóstico.
- El procesamiento corre en CPU y puede tardar varios minutos; el tiempo exacto
  depende de la imagen, los pasos, el equipo y las versiones instaladas.
- La demostración local no requiere GPU ni descarga modelos, pero solo suaviza
  la zona enmascarada; no realiza inpainting semántico.
- Los modelos de análisis pueden fallar o describir incorrectamente una foto
  histórica.
- El usuario proporciona la región y el prompt; la aplicación no detecta daños
  automáticamente.
- La imagen generada dentro de la máscara es contenido nuevo. Solo otra fuente
  histórica independiente podría verificar si coincide con lo que existía.

## 11. Archivos relacionados

| Archivo | Responsabilidad en el flujo |
|---|---|
| `app.py` | Interfaz Streamlit, carga, lienzo de máscara, parámetros, presentación y descarga |
| `restaurador/imagenes.py` | Preparación de imágenes, máscaras, composición y mapa de diferencias |
| `restaurador/modelos.py` | Identificadores/revisiones y caché de los modelos |
| `restaurador/procesamiento.py` | Orquestación de difusión, análisis y resultados |
| `tests/test_imagenes.py` | Pruebas unitarias de transformaciones sin descargar modelos |
| `pyproject.toml` y `uv.lock` | Dependencias y versiones reproducibles del entorno |
| `modelos_difusion/datos/LEEME.md` | Fuentes, descripción y criterios de los ejemplos fotográficos |
| `modelos_difusion/04_fotos_antiguas.ipynb` | Material de clase sobre máscaras e inpainting |
| `modelos_difusion/05_verificar_lo_reconstruido.ipynb` | Material de clase sobre mapas, SigLIP y RT-DETR |

Los notebooks son material de referencia para el aprendizaje; la aplicación
ejecutable sigue el flujo Python descrito en este documento.
