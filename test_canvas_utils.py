from __future__ import annotations

from PIL import Image

from canvas_utils import _data_url_to_image, _image_to_data_url, _resize_img


def test_image_to_data_url_roundtrip_preserves_dimensions() -> None:
    image = Image.new("RGB", (32, 24), color=(180, 108, 88))
    data_url = _image_to_data_url(image)
    restored = _data_url_to_image(data_url)

    assert data_url.startswith("data:image/png;base64,")
    assert restored.size == image.size


def test_resize_img_matches_requested_canvas_size() -> None:
    image = Image.new("RGB", (160, 90), color=(180, 108, 88))
    resized = _resize_img(image, new_height=300, new_width=480)
    assert resized.size == (480, 300)
