from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import streamlit.components.v1 as components
from PIL import Image


COMPONENT_BUILD_DIR = Path(__file__).resolve().parent / "vendor" / "drawable_canvas_build"
_component_func = components.declare_component("st_canvas_fixed", path=str(COMPONENT_BUILD_DIR))


@dataclass
class CanvasResult:
    image_data: np.ndarray | None = None
    json_data: dict[str, Any] | None = None


def _data_url_to_image(data_url: str) -> Image.Image:
    _, encoded = data_url.split(";base64,", 1)
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def _resize_img(img: Image.Image, new_height: int = 700, new_width: int = 700) -> Image.Image:
    h_ratio = new_height / img.height
    w_ratio = new_width / img.width
    return img.resize((int(img.width * w_ratio), int(img.height * h_ratio)))


def _image_to_data_url(image: Image.Image, output_format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.convert("RGBA").save(buffer, format=output_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{output_format.lower()};base64,{encoded}"


def st_canvas_fixed(
    fill_color: str = "#eee",
    stroke_width: int = 20,
    stroke_color: str = "black",
    background_color: str = "",
    background_image: Image.Image | None = None,
    update_streamlit: bool = True,
    height: int = 400,
    width: int = 600,
    drawing_mode: str = "freedraw",
    initial_drawing: dict[str, Any] | None = None,
    display_toolbar: bool = True,
    point_display_radius: int = 3,
    key: str | None = None,
) -> CanvasResult:
    """Patched drawable canvas that accepts embedded background images on web deploys."""

    background_image_url = None
    if background_image is not None:
        resized_background = _resize_img(background_image, height, width)
        background_image_url = _image_to_data_url(resized_background, output_format="PNG")
        background_color = ""

    cleaned_initial_drawing = {"version": "4.4.0"} if initial_drawing is None else dict(initial_drawing)
    cleaned_initial_drawing["background"] = background_color

    component_value = _component_func(
        fillColor=fill_color,
        strokeWidth=stroke_width,
        strokeColor=stroke_color,
        backgroundColor=background_color,
        backgroundImageURL=background_image_url,
        realtimeUpdateStreamlit=update_streamlit and (drawing_mode != "polygon"),
        canvasHeight=height,
        canvasWidth=width,
        drawingMode=drawing_mode,
        initialDrawing=cleaned_initial_drawing,
        displayToolbar=display_toolbar,
        displayRadius=point_display_radius,
        key=key,
        default=None,
    )

    if component_value is None:
        return CanvasResult()

    return CanvasResult(
        image_data=np.asarray(_data_url_to_image(component_value["data"])),
        json_data=component_value["raw"],
    )
