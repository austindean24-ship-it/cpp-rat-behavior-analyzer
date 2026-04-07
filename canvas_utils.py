from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image
from streamlit_drawable_canvas import CanvasResult, _component_func, _data_url_to_image, _resize_img


def _image_to_data_url(image: Image.Image, output_format: str = "JPEG") -> str:
    """Convert a Pillow image into an inline data URL.

    This avoids relying on Streamlit's internal media URL handling, which can be
    unreliable for `streamlit-drawable-canvas` background images on Community Cloud.
    """

    buffer = io.BytesIO()
    rgb_image = image.convert("RGB")
    if output_format.upper() == "JPEG":
        rgb_image.save(buffer, format="JPEG", quality=85, optimize=True)
        mime = "image/jpeg"
    else:
        rgb_image.save(buffer, format=output_format)
        mime = f"image/{output_format.lower()}"
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


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
    """Compatibility wrapper around `streamlit-drawable-canvas`.

    The package works well locally, but the hosted Streamlit deployment can fail
    to display the background image. This wrapper embeds the first frame directly
    into the component as a data URL instead of a temporary media-file URL.
    """

    background_image_url = None
    if background_image is not None:
        resized_background = _resize_img(background_image, height, width)
        background_image_url = _image_to_data_url(resized_background, output_format="JPEG")
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
        return CanvasResult

    return CanvasResult(
        np.asarray(_data_url_to_image(component_value["data"])),
        component_value["raw"],
    )
