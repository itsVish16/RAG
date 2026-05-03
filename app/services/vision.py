import base64
import httpx
from pathlib import Path
from app.core.config import settings


def encode_image_bytes_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def get_mime_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_map.get(ext, "image/png")


def extract_text_from_image(
    image_bytes: bytes,
    filename: str = "image.png",
    prompt: str = "Extract ALL text content from this image. Return only the extracted text, nothing else.",
) -> str:
    b64 = encode_image_bytes_to_base64(image_bytes)
    mime = get_mime_type(filename)

    response = httpx.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.mistral_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.mistral_vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            "max_tokens": 4096,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def extract_text_from_pdf_images(pdf_bytes: bytes) -> str:
    from pypdf import PdfReader
    import io

    reader = PdfReader(io.BytesIO(pdf_bytes))
    all_text = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            all_text.append(text)

        for image in page.images:
            try:
                image_text = extract_text_from_image(
                    image.data,
                    filename=image.name,
                    prompt=f"Extract ALL text from this image found on page {page_num + 1} of a PDF document.",
                )
                if image_text.strip():
                    all_text.append(f"[Image on page {page_num + 1}]: {image_text}")
            except Exception:
                continue

    return "\n\n".join(all_text)
