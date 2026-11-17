# Restauración fotográfica con difusión

MVP local y educativo para experimentar con la reparación de fotografías en blanco
y negro. La persona marca una región dañada; un modelo de difusión propone píxeles
para esa región y la aplicación muestra una comparación, un mapa de diferencias,
detecciones de objetos y descripciones generadas por IA.

> **Aviso:** una reconstrucción es una invención plausible del modelo, no una
> recuperación garantizada de la fotografía original. No debe utilizarse como
> evidencia histórica.

## Persona usuaria

**Estudiante terciario de 35 a 50 años**, con conocimientos de Python y modelos de
IA, que cursa Procesamiento Digital de Imágenes e Introducción a Visión por
Computadora. Necesita una aplicación local, comprensible y comentada para aprender
cómo se conectan una interfaz, una máscara, un modelo de difusión y modelos de
análisis visual.

## Qué puede hacer esta primera versión

- Cargar una foto propia o elegir un ejemplo local cuando esté disponible. Si
  la carpeta de ejemplos no está incluida, la app solicita cargar una foto.
- Convertirla a escala de grises y reducirla proporcionalmente para que su lado
  mayor no supere **512 píxeles**.
- Pintar en rojo una máscara manual. La máscara determina dónde puede proponer
  contenido el modelo.
- Ejecutar Stable Diffusion 1.5 Inpainting con semilla, instrucción, fuerza y
  cantidad de pasos configurables.
- Conservar exactamente, fuera de la máscara, los píxeles de la imagen de
  trabajo ya preparada. Esta imagen se convierte a escala de grises y puede
  reducirse a 512 px por lado; por eso el PNG no conserva el color ni siempre
  la resolución del archivo que se subió.
- Comparar original y resultado, visualizar la máscara y mostrar un mapa de
  diferencias.
- Ejecutar RT-DETR para detectar objetos y BLIP para describir la imagen antes y
  después.
- Probar una demostración local ligera, sin GPU ni descarga de modelos: aplica
  un filtro mediano dentro de la máscara y conserva el exterior. Es un suavizado
  para verificar el flujo de la app, no una reconstrucción inteligente.
- Descargar el resultado como PNG.

La aplicación presenta detecciones, captions y diferencias como indicios
orientativos. Ninguno prueba que los detalles reconstruidos sean verdaderos.

## Arquitectura

```text
Persona usuaria
    |
    v
Streamlit (app.py)
    |-- carga propia o imágenes de ejemplo
    |-- lienzo de máscara manual
    |-- modo demo local o pipeline completo de IA
    v
Procesamiento de imagen (restaurador/imagenes.py y procesamiento.py)
    |-- orientación EXIF, escala de grises, reducción a 512 px
    |-- conversión del trazo rojo a máscara binaria
    v
Procesamiento según el modo
    |-- demo local: filtro mediano, sin modelos
    |-- IA: Stable Diffusion 1.5 Inpainting, ejecución CPU
    |-- composición que conserva intacto el exterior de la máscara
    v
Análisis visual (solo en el pipeline completo)
    |-- RT-DETR: detección de objetos
    |-- BLIP: caption de imagen
    v
Comparación y exportación
    |-- original, reconstrucción, máscara y mapa de diferencias
    |-- explicación de límites y descarga PNG
```

### Módulos

- `app.py`: interfaz, entrada de datos, controles y presentación educativa de los
  resultados.
- `restaurador/imagenes.py`: operaciones deterministas de imágenes y máscaras; no
  carga modelos.
- `restaurador/modelos.py`: identificadores y revisiones fijadas, y carga
  cacheada de modelos en CPU.
- `restaurador/procesamiento.py`: orquesta inpainting, análisis y comparación.
- `tests/test_imagenes.py`: pruebas de resolución, máscaras, composición y
  diferencias sin descargar modelos.
- `tests/test_procesamiento.py`: pruebas del modo local sin cargar modelos.

Los módulos y bloques principales incluyen comentarios y docstrings en español
para que el flujo sea útil como material de aprendizaje.

Para seguir en detalle cada etapa desde la carga de la imagen hasta la descarga,
consultá [`pipeline.md`](pipeline.md).

## Tecnologías y modelos

| Componente | Identificador y revisión fijada | Licencia declarada en el Hub | Función |
|---|---|---|---|
| Inpainting | `stable-diffusion-v1-5/stable-diffusion-inpainting` — `8a4288a76071f7280aedbdb3253bdb9e9d5d84bb` | CreativeML OpenRAIL-M | Proponer contenido en la región enmascarada |
| Detector | `PekingU/rtdetr_r50vd` — `df939e661d8c52e80608d1ec566561aabd25a4e7` | Apache-2.0 | Comparar objetos detectados antes y después |
| Caption de imagen | `Salesforce/blip-image-captioning-base` — `82a37760796d32b1411fe092ab5d4e227313294b` | BSD-3-Clause | Generar descripciones generales en inglés |

Las revisiones fijan las versiones de los pesos consultadas en Hugging Face Hub.
Las dependencias Python se registran en `uv.lock`. Antes de redistribuir pesos,
revisá las condiciones completas de cada licencia y las fichas enlazadas del Hub.

### Stack de software y dependencias

La aplicación es una interfaz local en **Streamlit**, con lienzo de dibujo
mediante **streamlit-drawable-canvas**. El procesamiento usa **Pillow** y
**NumPy**; el inpainting se ejecuta con **Diffusers** y **PyTorch**; **Transformers**
provee los pipelines de detección y descripción; **Accelerate** es dependencia
de Diffusers. **uv** instala y sincroniza el entorno desde `pyproject.toml` y
`uv.lock`. Las pruebas se ejecutan con **pytest**.

`pyproject.toml` define los rangos permitidos y `uv.lock` registra las versiones
resueltas para este proyecto. La tabla distingue ambos datos; no afirma que sean
las últimas versiones publicadas.

| Componente | Rango declarado | Versión en `uv.lock` | Uso |
|---|---:|---:|---|
| Python | `>=3.12,<3.13` | `3.12` | Runtime del proyecto |
| Streamlit | `>=1.38,<1.41` | `1.40.2` | Interfaz web local |
| streamlit-drawable-canvas | `>=0.9.3,<0.10` | `0.9.3` | Lienzo para dibujar la máscara |
| PyTorch (`torch`) | `>=2.4,<3` | `2.15.0` | Tensores e inferencia; el código usa CPU |
| Diffusers | `>=0.34,<1` | `0.39.0` | Pipeline de Stable Diffusion Inpainting |
| Transformers | `>=4.45,<5` | `4.57.6` | RT-DETR y BLIP |
| Accelerate | `>=1.1,<2` | `1.15.0` | Dependencia de Diffusers |
| NumPy | `>=1.26,<3` | `2.5.4` | Máscaras y operaciones de píxeles |
| Pillow | `>=10,<13` | `11.3.0` | Lectura, transformación y exportación de imágenes |
| pytest (desarrollo) | `>=8,<9` | `8.4.2` | Pruebas automatizadas |

### Hardware y aceleración

Equipo de desarrollo informado: **Intel Core i7-1365U, Intel UHD integrada y
16 GB de RAM**. Los 8 GB informados para la UHD son memoria compartida, no 8 GB de
VRAM dedicada.

Esta versión usa **CPU con PyTorch** como línea base de compatibilidad; no usa CUDA.
La demostración local con filtro mediano no carga modelos y no requiere GPU ni
conexión. Para el pipeline completo, la primera descarga de modelos requiere
Internet; las ejecuciones posteriores reutilizan el caché local de Hugging Face.
En CPU, el proceso completo puede tardar varios minutos y consumir bastante RAM.
La aceleración OpenVINO y el tiempo real de inferencia aún no se midieron en este
equipo.

## Instalación y ejecución

Requisitos: Windows, Python 3.12, `uv` y conexión a Internet para descargar
dependencias y modelos en la primera ejecución.

Desde la carpeta del proyecto:

```powershell
uv sync
uv run python -m streamlit run app.py --server.address 127.0.0.1
```

En Windows también podés iniciar la aplicación con doble clic en
[`ejecutar_app.bat`](ejecutar_app.bat). Ese archivo cambia a la carpeta del
proyecto y ejecuta Streamlit usando `uv`; la consola que se abre debe permanecer
abierta mientras uses la aplicación. Si el lanzador `streamlit.exe` produce
`ModuleNotFoundError`, usá el comando `python -m streamlit` indicado arriba, que
evita ese lanzador. Para detener la aplicación, presioná `Ctrl+C` en la ventana
de la terminal.

Streamlit mostrará la dirección local en la terminal y abrirá la aplicación en el
navegador. Para ejecutar las pruebas unitarias:

```powershell
uv run pytest
```

## Uso

1. Cargá una imagen JPG/PNG o seleccioná un ejemplo local si está disponible.
2. Pintá en rojo el rayón o la zona que querés procesar.
3. Seleccioná **Demostración local (sin IA)** para probar sin modelos: aplica un
   filtro mediano a la zona marcada. No reconstruye contenido ni necesita
   conexión para descargar pesos.
4. Si querés ejecutar los modelos, seleccioná **Pipeline completo de IA (CPU)**.
   Escribí una instrucción breve y elegí pasos, fuerza y semilla. La instrucción
   guía la generación, pero no es evidencia sobre el contenido original. Para
   una primera prueba en CPU, empezá con 5 pasos; puede tardar varios minutos.
5. Presioná el botón de procesamiento. Compará las imágenes, la máscara y el
   mapa de diferencias; en el pipeline completo también se muestran captions y
   detecciones. Descargá el PNG si lo necesitás.

Los captions de BLIP se muestran en inglés, que es el idioma producido por el
modelo elegido. No se traducen para evitar atribuir al modelo una traducción que
no realizó.

## Ejemplos y capturas

La app busca los ejemplos `danado_retrato_1860.jpg` y
`danado_puente_1888.jpg` en `modelos_difusion/datos/`. Esa carpeta no está
incluida en esta versión del proyecto, así que no hay ejemplos disponibles y la
app solicita subir una fotografía propia. Si se agregan ejemplos en una versión
futura, documentar su fuente y descripción junto con los archivos.

Las capturas de la interfaz se incorporarán después de ejecutar esta versión en
el equipo objetivo; todavía no hay una captura verificada de una ejecución local.

## Decisiones de diseño desde HAI

- **Control de la persona:** la persona marca la máscara, ajusta la instrucción y
  controla fuerza, pasos y semilla.
- **Visibilidad de la intervención:** la máscara se muestra en rojo; el mapa de
  diferencias indica dónde cambiaron los píxeles.
- **Límites explícitos:** la interfaz distingue una propuesta generada de una
  recuperación verdadera. Detectores y captions no se presentan como jueces de
  autenticidad.
- **Cambios acotados:** aunque la tubería genere alteraciones fuera de la zona,
  la composición final conserva exactamente, fuera de la máscara, los píxeles
  de la imagen de trabajo ya convertida a gris y redimensionada.
- **Reproducibilidad:** se fijan las revisiones de los modelos y se expone la
  semilla del generador.
- **Menos sorpresa:** se informa que las imágenes se pasan a escala de grises y
  que se reduce la resolución para el procesamiento.

## Qué es real y qué es inventado

- **Conservado:** los píxeles no enmascarados de la imagen de trabajo que la app
  preparó para el procesamiento; no necesariamente coinciden con el archivo
  original en color o resolución.
- **Generado:** los píxeles dentro de la máscara final; la difusión propone una
  reconstrucción condicionada por el contexto y la instrucción.
- **Creíble:** algo que parece compatible con la fotografía. Que parezca creíble
  no significa que haya ocurrido.
- **Verificable:** solo puede establecerse con evidencia externa confiable, como
  otra copia de la fotografía o documentación histórica.
- **Qué muestran las herramientas:** el mapa muestra diferencias de píxeles;
  RT-DETR detecta categorías de objetos conocidas y BLIP describe contenido
  general. Ninguno puede confirmar los detalles inventados.

## Limitaciones conocidas

- La ejecución en CPU puede ser lenta. OpenVINO no está integrado ni validado en
  el hardware disponible.
- La demostración local solo suaviza los píxeles marcados con un filtro mediano;
  no reemplaza la reconstrucción generativa ni el análisis visual del pipeline
  completo.
- El primer arranque descarga varios modelos, que pueden ocupar varios
  gigabytes en disco.
- El resultado trabaja hasta 512 px; no recupera detalle de alta resolución.
- El usuario debe marcar el daño manualmente y los bordes de la máscara pueden
  requerir ajustes.
- Aunque se conserva el exterior de la máscara, la región interior puede cambiar
  de manera plausible pero incorrecta.
- RT-DETR puede no reconocer objetos de fotografías históricas, pequeñas o
  dañadas. Una detección no demuestra que un objeto se haya preservado.
- BLIP produce captions en inglés, puede omitir detalles y no compara ni valida
  fielmente las dos versiones.
- La primera versión no incluye detección automática del daño, restauración por
  lotes ni métricas de autenticidad.

## Uso de agentes de IA

La persona usuaria definió el problema, el alcance educativo, la interfaz
Streamlit, la resolución de procesamiento y el hardware. El asistente propuso la
separación modular, la composición fuera de máscara, las revisiones fijadas de
los modelos y las pruebas unitarias. Las decisiones propuestas se revisan con
evidencia y se explican en este documento.
