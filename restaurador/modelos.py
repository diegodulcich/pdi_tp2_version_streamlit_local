"""Carga bajo demanda de los modelos de difusión y análisis de Hugging Face."""

from __future__ import annotations

import os

import streamlit as st

MODELO_INPAINTING = "stable-diffusion-v1-5/stable-diffusion-inpainting"
REVISION_INPAINTING = "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb"
MODELO_DETECCION = "PekingU/rtdetr_r50vd"
REVISION_DETECCION = "df939e661d8c52e80608d1ec566561aabd25a4e7"
MODELO_DESCRIPCION = "Salesforce/blip-image-captioning-base"
REVISION_DESCRIPCION = "82a37760796d32b1411fe092ab5d4e227313294b"


def _limitar_hilos_cpu() -> None:
    """Evita que la inferencia ocupe todos los hilos del equipo por defecto."""
    import torch

    hilos = max(1, min(4, os.cpu_count() or 1))
    torch.set_num_threads(hilos)


@st.cache_resource(show_spinner=False)
def cargar_inpainting():
    """Carga Stable Diffusion 1.5 Inpainting una vez y lo ejecuta en CPU/FP32.

    La Intel UHD no ofrece CUDA; CPU es la ruta de compatibilidad para esta
    primera versión. OpenVINO se evaluará aparte para no presentar aceleración
    no probada como si ya estuviera disponible.
    """
    import torch
    from diffusers import AutoPipelineForInpainting

    _limitar_hilos_cpu()
    tuberia = AutoPipelineForInpainting.from_pretrained(
        MODELO_INPAINTING,
        revision=REVISION_INPAINTING,
        torch_dtype=torch.float32,
    )
    tuberia = tuberia.to("cpu")
    tuberia.enable_attention_slicing()
    return tuberia


@st.cache_resource(show_spinner=False)
def cargar_detector():
    """Carga RT-DETR una vez para comparar objetos antes y después."""
    from transformers import pipeline

    return pipeline(
        "object-detection",
        model=MODELO_DETECCION,
        revision=REVISION_DETECCION,
        device=-1,
    )


@st.cache_resource(show_spinner=False)
def cargar_descriptor():
    """Carga BLIP una vez para generar descripciones de las imágenes."""
    from transformers import pipeline

    return pipeline(
        "image-to-text",
        model=MODELO_DESCRIPCION,
        revision=REVISION_DESCRIPCION,
        device=-1,
    )
