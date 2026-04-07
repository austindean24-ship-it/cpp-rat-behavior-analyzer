from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import CanvasResult, _component_func, _data_url_to_image, _resize_img


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
CANVAS_BACKGROUND_DIR = STATIC_DIR / "canvas_backgrounds"
CANVAS_BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_base_url_path(base_url_path: str | None = None) -> str:
    """Return Streamlit's configured base path in a URL-safe form."""

    if base_url_path is None:
        try:
            base_url_path = st._config.get_option("server.baseUrlPath")
        except Exception:  # noqa: BLE001
            base_url_path = ""
    normalized = (base_url_path or "").strip()
    if normalized and not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return normalized.rstrip("/")


def _build_static_background_url(filename: str, base_url_path: str | None = None) -> str:
    """Build the URL expected by the drawable-canvas frontend.

    The component prepends the Streamlit app origin on the frontend, so this must
    be a root-relative path such as `/app/static/...`.
    """

    base_path = _normalize_base_url_path(base_url_path)
    return f"{base_path}/app/static/canvas_backgrounds/{filename}"


def _write_background_image_to_static(
    image: Image.Image,
    output_dir: Path = CANVAS_BACKGROUND_DIR,
    base_url_path: str | None = None,
) -> str:
    """Persist the canvas background image into Streamlit's static directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_image = image.convert("RGB")
    image_hash = hashlib.sha256(rgb_image.tobytes()).hexdigest()[:20]
    filename = f"canvas_bg_{image_hash}_{rgb_image.width}x{rgb_image.height}.jpg"
    output_path = output_dir / filename
    if not output_path.exists():
        rgb_image.save(output_path, format="JPEG", quality=90, optimize=True)
    return _build_static_background_url(filename, base_url_path=base_url_path)


def _streamlit_media_url(image: Image.Image, width: int, key: str | None = None) -> str:
    """Fallback to Streamlit's media-file manager if static-file writing fails."""

    from streamlit.elements.lib.image_utils import image_to_url
    from streamlit.elements.lib.layout_utils import LayoutConfig

    image_id = f"drawable-canvas-bg-{hashlib.md5(image.tobytes()).hexdigest()}-{key or 'canvas'}"
    media_url = image_to_url(
        image,
        LayoutConfig(width=width),
        True,
        "RGB",
        "PNG",
        image_id,
    )
    return f"{_normalize_base_url_path()}{media_url}"


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

    The packaged component expects a root-relative URL for its background image.
    On Streamlit Community Cloud, inline data URLs can fail because the frontend
    blindly prepends the app origin. Writing the frame into `/app/static/...`
    keeps local and deployed behavior aligned.
    """

    background_image_url = None
    if background_image is not None:
        resized_background = _resize_img(background_image, height, width)
        try:
            background_image_url = _write_background_image_to_static(resized_background)
        except Exception:  # noqa: BLE001
            background_image_url = _streamlit_media_url(resized_background, width=width, key=key)
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
