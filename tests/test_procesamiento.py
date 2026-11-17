"""Pruebas del procesamiento que no requieren descargar modelos."""

import pytest
from PIL import Image

from restaurador import procesamiento
from restaurador.procesamiento import ejecutar_pipeline_basico, reparar_imagen_basica


def test_reparacion_basica_suaviza_la_mascara_y_conserva_el_exterior() -> None:
    """El filtro mediano opera en la máscara sin modificar los demás píxeles."""
    imagen = Image.new("RGB", (9, 9), (120, 120, 120))
    imagen.putpixel((4, 4), (0, 0, 0))
    mascara = Image.new("L", imagen.size, 0)
    mascara.paste(255, (3, 3, 6, 6))

    resultado = reparar_imagen_basica(imagen, mascara)

    assert resultado.getpixel((4, 4)) == (120, 120, 120)
    assert resultado.getpixel((0, 0)) == imagen.getpixel((0, 0))


def test_pipeline_basico_devuelve_comparacion_sin_cargar_modelos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La ruta local genera comparación sin ejecutar detección ni captioning."""
    imagen = Image.new("RGB", (8, 8), (120, 120, 120))
    mascara = Image.new("L", imagen.size, 0)
    mascara.paste(255, (2, 2, 6, 6))

    def modelo_no_esperado() -> None:
        raise AssertionError("El modo básico no debe cargar modelos.")

    monkeypatch.setattr(procesamiento, "cargar_inpainting", modelo_no_esperado)
    monkeypatch.setattr(procesamiento, "cargar_detector", modelo_no_esperado)
    monkeypatch.setattr(procesamiento, "cargar_descriptor", modelo_no_esperado)

    resultado = ejecutar_pipeline_basico(imagen, mascara)

    assert resultado.imagen_reparada.size == imagen.size
    assert resultado.mapa_diferencias.size == imagen.size
    assert resultado.detecciones_originales == []
    assert resultado.detecciones_reparadas == []
    assert resultado.descripcion_original == ""
    assert resultado.descripcion_reparada == ""


def test_reparacion_basica_requiere_mascara_no_vacia() -> None:
    """El modo local comunica si se intenta procesar sin marcar una zona."""
    imagen = Image.new("RGB", (8, 8), (120, 120, 120))
    mascara = Image.new("L", imagen.size, 0)

    with pytest.raises(ValueError, match="zona que querés procesar"):
        reparar_imagen_basica(imagen, mascara)
