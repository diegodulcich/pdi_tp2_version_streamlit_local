"""Pruebas de las operaciones de imagen que no requieren descargar modelos."""

import numpy as np
from PIL import Image

from restaurador.imagenes import (
    componer_solo_mascara,
    crear_mapa_diferencias,
    extraer_mascara_canvas,
    normalizar_mascara,
    preparar_imagen,
)


def test_preparar_imagen_convierte_a_gris_y_respeta_limite() -> None:
    """La preparación devuelve RGB monocromático y no supera 512 píxeles."""
    imagen = Image.new("RGB", (1200, 800), (200, 80, 10))

    preparada = preparar_imagen(imagen)

    assert preparada.mode == "RGB"
    assert max(preparada.size) <= 512
    assert preparada.getpixel((0, 0))[0] == preparada.getpixel((0, 0))[1]
    assert preparada.getpixel((0, 0))[1] == preparada.getpixel((0, 0))[2]
    assert all(lado % 8 == 0 for lado in preparada.size)


def test_extraer_mascara_canvas_identifica_trazo_rojo() -> None:
    """El trazo rojo del lienzo se convierte en blanco sobre máscara negra."""
    lienzo = np.zeros((12, 12, 4), dtype=np.uint8)
    lienzo[:, :, 3] = 255
    lienzo[2:5, 3:7, :3] = (255, 0, 0)

    mascara = np.asarray(extraer_mascara_canvas(lienzo))

    assert mascara[3, 4] == 255
    assert mascara[0, 0] == 0


def test_componer_solo_mascara_conserva_el_exterior() -> None:
    """La composición usa los píxeles nuevos solo en la región marcada."""
    original = Image.new("RGB", (8, 8), (20, 20, 20))
    generada = Image.new("RGB", (8, 8), (200, 100, 50))
    mascara = Image.new("L", (8, 8), 0)
    mascara.paste(255, (2, 2, 6, 6))

    resultado = componer_solo_mascara(original, generada, mascara)

    assert resultado.getpixel((0, 0)) == original.getpixel((0, 0))
    assert resultado.getpixel((3, 3)) == (124, 124, 124)


def test_mascara_se_ajusta_con_vecino_mas_cercano() -> None:
    """El redimensionamiento de máscaras binarias no crea grises intermedios."""
    mascara = Image.new("L", (2, 2), 0)
    mascara.putpixel((1, 1), 255)

    normalizada = normalizar_mascara(mascara, (8, 8))

    assert set(np.unique(np.asarray(normalizada))) <= {0, 255}


def test_mapa_diferencias_detecta_cambio_y_mismo_tamano() -> None:
    """El mapa compara imágenes alineadas y reporta el área alterada."""
    original = Image.new("RGB", (8, 8), (0, 0, 0))
    resultado = original.copy()
    resultado.paste((255, 255, 255), (0, 0, 4, 8))

    mapa, porcentaje = crear_mapa_diferencias(original, resultado)

    assert mapa.size == original.size
    assert porcentaje == 50.0
