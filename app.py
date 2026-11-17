"""Interfaz Streamlit del MVP educativo de restauración con difusión."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError
from streamlit_drawable_canvas import st_canvas

from restaurador.imagenes import (
    extraer_mascara_canvas,
    preparar_imagen,
    superponer_mascara,
)
from restaurador.procesamiento import (
    ResultadoPipeline,
    ejecutar_pipeline,
    ejecutar_pipeline_basico,
)

CARPETA_EJEMPLOS = Path(__file__).parent / "modelos_difusion" / "datos"
EJEMPLOS = {
    "Retrato dañado (1860)": CARPETA_EJEMPLOS / "danado_retrato_1860.jpg",
    "Puente dañado (1888)": CARPETA_EJEMPLOS / "danado_puente_1888.jpg",
}

st.set_page_config(page_title="Restauración educativa con IA", layout="wide")


def leer_imagen_seleccionada() -> tuple[Image.Image, str]:
    """Lee la carga de la persona usuaria o el ejemplo seleccionado."""
    archivo = st.file_uploader(
        "Subí una fotografía propia (JPG o PNG)",
        type=["jpg", "jpeg", "png"],
        help="Para este MVP, la imagen se convertirá a blanco y negro y se procesará hasta 512 px.",
    )
    ejemplos_disponibles = {
        nombre: ruta for nombre, ruta in EJEMPLOS.items() if ruta.is_file()
    }
    ejemplo = (
        st.selectbox("O elegí una imagen de ejemplo", list(ejemplos_disponibles))
        if ejemplos_disponibles
        else None
    )

    try:
        if archivo is not None:
            bytes_imagen = archivo.getvalue()
            imagen = Image.open(io.BytesIO(bytes_imagen))
            nombre = archivo.name
        elif ejemplo is not None:
            ruta = ejemplos_disponibles[ejemplo]
            bytes_imagen = ruta.read_bytes()
            imagen = Image.open(io.BytesIO(bytes_imagen))
            nombre = ruta.name
        else:
            st.info("No hay ejemplos locales disponibles. Subí una fotografía para continuar.")
            st.stop()
    except (OSError, UnidentifiedImageError, ValueError) as error:
        st.error(f"No se pudo abrir la imagen seleccionada: {error}")
        st.stop()

    return preparar_imagen(imagen), nombre


def clave_solicitud(
    imagen: Image.Image,
    mascara: Image.Image,
    modo: str,
    prompt: str,
    semilla: int,
    pasos: int,
    fuerza: float,
) -> str:
    """Identifica los parámetros exactos para no mostrar resultados desactualizados."""
    resumen = (
        f"{modo}|{prompt}|{semilla}|{pasos}|{fuerza}|"
        f"{imagen.size}|{hashlib.sha256(imagen.tobytes()).hexdigest()}|"
        f"{hashlib.sha256(mascara.tobytes()).hexdigest()}"
    )
    return hashlib.sha256(resumen.encode("utf-8")).hexdigest()


def mostrar_detecciones(titulo: str, detecciones: list[dict]) -> None:
    """Presenta etiquetas y puntajes sin tratarlos como verificación de identidad."""
    st.markdown(f"**{titulo}**")
    confiables = [item for item in detecciones if item["score"] >= 0.5]
    if not confiables:
        st.write("No se detectaron objetos con puntaje igual o mayor a 0,50.")
        return
    for item in confiables:
        st.write(f"- {item['label']}: {item['score']:.2f}")


st.title("Restauración fotográfica con difusión")
st.caption("MVP local y educativo para experimentar con una zona marcada.")
st.warning(
    "En el modo de IA, lo reconstruido es una invención plausible del modelo, "
    "no una recuperación garantizada de la foto ni evidencia histórica."
)
st.info(
    "No hace falta una GPU para abrir la app. La demostración local funciona sin "
    "modelos; el pipeline completo de IA también puede ejecutarse en CPU, pero "
    "puede tardar varios minutos y consumir bastante memoria."
)

# Bloque 1: carga y preparación. Toda imagen se orienta, pasa a gris y se limita
# a 512 px antes de mostrarla o enviarla a los modelos.
try:
    imagen, nombre_imagen = leer_imagen_seleccionada()
except (OSError, UnidentifiedImageError, ValueError) as error:
    st.error(f"No se pudo preparar la imagen: {error}")
    st.stop()

st.subheader("1. Elegí la zona a procesar")
st.write(
    "Pintá en rojo sobre el área dañada. La máscara es una decisión de la persona "
    "usuaria: blanco indica la zona que se procesará y negro la que se conservará."
)
columna_imagen, columna_lienzo = st.columns(2)
with columna_imagen:
    st.image(imagen, caption=f"Imagen de trabajo: {nombre_imagen}", use_container_width=True)
with columna_lienzo:
    lienzo = st_canvas(
        fill_color="rgba(255, 0, 0, 0.25)",
        stroke_width=14,
        stroke_color="#ff0000",
        background_image=imagen,
        background_color="#ffffff",
        drawing_mode="freedraw",
        height=imagen.height,
        width=imagen.width,
        update_streamlit=True,
        display_toolbar=True,
        key=f"mascara-{hashlib.sha256(imagen.tobytes()).hexdigest()[:12]}",
    )

if lienzo.image_data is None:
    mascara = Image.new("L", imagen.size, 0)
else:
    mascara = extraer_mascara_canvas(lienzo.image_data)

st.image(
    superponer_mascara(imagen, mascara),
    caption="Vista previa de la máscara: la zona roja se procesará.",
    use_container_width=True,
)

st.subheader("2. Elegí el modo de procesamiento")
modo = st.radio(
    "Modo",
    ["Demostración local (sin IA)", "Pipeline completo de IA (CPU)"],
    help=(
        "La demostración usa un filtro local y no descarga modelos. El pipeline "
        "completo carga difusión, detección y captioning en CPU."
    ),
)
modo_basico = modo == "Demostración local (sin IA)"

if modo_basico:
    st.subheader("3. Demostración local")
    prompt, pasos, fuerza, semilla = "", 5, 0.85, 42
    st.info(
        "Esta demostración aplica un filtro mediano dentro de la máscara. "
        "Sirve para probar el flujo local, pero no usa IA ni reconstruye "
        "contenido como el modelo de difusión."
    )
else:
    # Bloque 2: controles reproducibles para el pipeline de difusión.
    st.subheader("3. Configurá el experimento")
    prompt = st.text_input(
        "Instrucción para la región reconstruida",
        value="black and white archival photograph, natural photographic texture",
        max_chars=200,
        help="La instrucción guía la generación; no aporta evidencia sobre cómo era la foto.",
    )
    col_pasos, col_fuerza, col_semilla = st.columns(3)
    with col_pasos:
        pasos = st.slider(
            "Pasos de difusión",
            min_value=5,
            max_value=20,
            value=5,
            help=(
                "Cada paso es una iteración de generación del modelo. Más pasos "
                "suelen tardar más y pueden refinar la propuesta, pero no hacen que "
                "sea más verdadera. En CPU, empezá con el mínimo."
            ),
        )
    with col_fuerza:
        fuerza = st.slider(
            "Fuerza de reconstrucción",
            min_value=0.3,
            max_value=1.0,
            value=0.85,
            step=0.05,
            help="Un valor alto permite mayor libertad para generar dentro de la máscara.",
        )
    with col_semilla:
        semilla = st.number_input(
            "Semilla reproducible",
            min_value=0,
            max_value=999_999,
            value=42,
        )

# Bloque 3: ejecución bajo demanda. La carga de los modelos está cacheada; la
# inferencia corre solo cuando se presiona el botón, no en cada cambio de control.
texto_boton = "Probar en local" if modo_basico else "Reconstruir y analizar"
if st.button(texto_boton, type="primary"):
    solicitud = clave_solicitud(
        imagen,
        mascara,
        modo,
        prompt,
        int(semilla),
        pasos,
        fuerza,
    )
    try:
        if modo_basico:
            with st.spinner("Aplicando el filtro local..."):
                resultado = ejecutar_pipeline_basico(imagen=imagen, mascara=mascara)
        else:
            with st.spinner(
                "Procesando en CPU: primero difusión y luego análisis visual. "
                "La primera ejecución también descarga los modelos."
            ):
                resultado = ejecutar_pipeline(
                    imagen=imagen,
                    mascara=mascara,
                    prompt=prompt,
                    semilla=int(semilla),
                    pasos=pasos,
                    fuerza=fuerza,
                )
        st.session_state["resultado"] = resultado
        st.session_state["solicitud"] = solicitud
        st.success("Proceso terminado.")
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        st.error(f"No se pudo completar el procesamiento: {error}")
        st.exception(error)

# Bloque 4: resultados y límites. Las detecciones y las descripciones son
# indicios automáticos; ni el mapa ni los modelos certifican que la foto sea fiel.
resultado_guardado = st.session_state.get("resultado")
solicitud_actual = clave_solicitud(
    imagen,
    mascara,
    modo,
    prompt,
    int(semilla),
    pasos,
    fuerza,
)
if isinstance(resultado_guardado, ResultadoPipeline):
    if st.session_state.get("solicitud") != solicitud_actual:
        st.info("Cambiaste la imagen o los parámetros. Volvé a ejecutar para actualizar el resultado.")
    else:
        st.subheader("4. Comparación")
        original_columna, reparada_columna = st.columns(2)
        with original_columna:
            st.image(resultado_guardado.imagen_original, caption="Antes", use_container_width=True)
        with reparada_columna:
            etiqueta_resultado = (
                "Después · suavizado local"
                if modo_basico
                else "Después · propuesta de IA"
            )
            st.image(
                resultado_guardado.imagen_reparada,
                caption=etiqueta_resultado,
                use_container_width=True,
            )

        st.image(
            resultado_guardado.mapa_diferencias,
            caption=(
                "Mapa de diferencias: tonos claros indican más cambio. "
                f"Cambió notablemente el {resultado_guardado.porcentaje_cambiado:.1f}% "
                "de los píxeles (umbral 10/255)."
            ),
            use_container_width=True,
        )

        if modo_basico:
            st.info(
                "La demostración local no ejecuta análisis de objetos ni genera "
                "descripciones. Seleccioná el pipeline completo para esas funciones."
            )
        else:
            st.subheader("5. Análisis visual automático")
            st.write(
                "BLIP genera descripciones en inglés. Se muestran tal cual para no "
                "presentar una traducción automática como si la hubiera producido el modelo."
            )
            antes_columna, despues_columna = st.columns(2)
            with antes_columna:
                st.markdown("**Descripción de la imagen original**")
                st.write(resultado_guardado.descripcion_original or "El modelo no generó una descripción.")
                mostrar_detecciones("Objetos detectados", resultado_guardado.detecciones_originales)
            with despues_columna:
                st.markdown("**Descripción de la propuesta reconstruida**")
                st.write(resultado_guardado.descripcion_reparada or "El modelo no generó una descripción.")
                mostrar_detecciones("Objetos detectados", resultado_guardado.detecciones_reparadas)

        st.subheader("Qué hizo el sistema")
        if modo_basico:
            st.write(
                "Aplicó un filtro mediano dentro de la máscara y conservó "
                "exactamente los píxeles externos. El suavizado puede disimular "
                "algunos defectos pequeños, pero no inventa ni recupera el contenido."
            )
        else:
            st.write(
                "Usó la instrucción y el contexto alrededor de la máscara para generar "
                "una propuesta con difusión. Después conservó exactamente los píxeles "
                "fuera de la máscara. El mapa muestra cambios de píxeles; RT-DETR y "
                "BLIP describen contenido general, pero ninguno puede verificar los "
                "detalles inventados dentro del daño."
            )
        bytes_salida = io.BytesIO()
        resultado_guardado.imagen_reparada.save(bytes_salida, format="PNG")
        st.download_button(
            "Descargar resultado (PNG)",
            data=bytes_salida.getvalue(),
            file_name=(
                "resultado_local.png"
                if modo_basico
                else "reconstruccion_ia.png"
            ),
            mime="image/png",
        )
