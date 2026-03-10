"""
GLM-OCR backend using Zhipu AI API.

This module provides OCR functionality using Zhipu AI's GLM-4V vision model.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from notely.config import OCRConfig
from notely.ocr.base import OCRBackend, OCRResult, TextBlock

logger = logging.getLogger(__name__)


class GLMOCRBackend(OCRBackend):
    """
    OCR backend using Zhipu AI GLM-4V vision model.

    This backend uses the GLM-4V multimodal model to perform OCR on images
    and PDF files via the Zhipu AI API.

    Attributes:
        config: OCR configuration instance.
        model: Model identifier (e.g., 'glm-4v-flash').
        client: ZhipuAI client instance.
    """

    def __init__(self, config: OCRConfig) -> None:
        """
        Initialize GLM-OCR backend.

        Args:
            config: OCR configuration instance.

        Raises:
            ValueError: If API key is not provided in config.
            ImportError: If zhipuai package is not installed.
        """
        if not config.api_key:
            raise ValueError("API key required for GLM-OCR backend")

        self.config = config
        self.model = config.model

        # Lazy import to avoid dependency issues
        try:
            from zhipuai import ZhipuAI  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "zhipuai package is required for GLM-OCR. Install it with: uv add zhipuai"
            ) from e

        self.client = ZhipuAI(api_key=config.api_key, base_url=config.base_url)
        logger.info(f"Initialized GLM-OCR backend with model: {self.model}")

    def _encode_file(self, file_path: Path | str) -> str:
        """
        Encode a file to base64 string.

        Args:
            file_path: Path to the file.

        Returns:
            Base64 encoded string of the file content.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _call_api(self, image_base64: str, file_type: str = "image") -> str:
        """
        Call GLM-4V API to extract text from image.

        Args:
            image_base64: Base64 encoded image data.
            file_type: Type of file ('image' or 'pdf').

        Returns:
            Extracted text content.

        Raises:
            RuntimeError: If API call fails.
        """
        try:
            # Determine MIME type based on file type
            if file_type == "pdf":
                mime_type = "application/pdf"
            else:
                # Default to JPEG for images
                mime_type = "image/jpeg"

            # Construct the image URL with base64 data
            image_url = f"data:{mime_type};base64,{image_base64}"

            # Call GLM-4V API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {
                                "type": "text",
                                "text": "请识别并提取图片中的所有文字内容，保持原有的格式和结构。",
                            },
                        ],
                    }
                ],
            )

            content = response.choices[0].message.content or ""
            return content

        except Exception as e:
            logger.error(f"GLM-OCR API call failed: {e}")
            raise RuntimeError(f"GLM-OCR API call failed: {e}") from e

    def recognize(self, image_path: Path | str) -> OCRResult:
        """
        Recognize text in an image.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult containing detected text blocks.

        Raises:
            FileNotFoundError: If the image file does not exist.
            RuntimeError: If API call fails.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        logger.debug(f"Recognizing text in image: {image_path}")

        # Encode image to base64
        image_base64 = self._encode_file(image_path)

        # Call API
        text_content = self._call_api(image_base64, file_type="image")

        # Create TextBlock from result
        text_block = TextBlock(
            text=text_content,
            confidence=1.0,  # GLM-4V doesn't provide confidence scores
            bbox=(0, 0, 0, 0),  # No bbox info from API
            block_type="text",
        )

        return OCRResult(
            text_blocks=[text_block],
            source_path=str(image_path),
            metadata={"model": self.model, "provider": "zhipu"},
        )

    def recognize_pdf(self, pdf_path: Path | str) -> list[OCRResult]:
        """
        Recognize text in a PDF file.

        This method sends the entire PDF to the GLM-4V API for processing.
        Note: GLM-4V processes the PDF as a whole, so we return a single
        OCRResult with page_number set to None.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of OCRResult, one per detected page or section.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            RuntimeError: If API call fails.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.debug(f"Recognizing text in PDF: {pdf_path}")

        # Encode PDF to base64
        pdf_base64 = self._encode_file(pdf_path)

        # Call API
        text_content = self._call_api(pdf_base64, file_type="pdf")

        # Create TextBlock from result
        text_block = TextBlock(
            text=text_content,
            confidence=1.0,  # GLM-4V doesn't provide confidence scores
            bbox=(0, 0, 0, 0),  # No bbox info from API
            block_type="text",
        )

        # Return single result for the entire PDF
        # In the future, we could split by page if the API supports it
        result = OCRResult(
            text_blocks=[text_block],
            source_path=str(pdf_path),
            page_number=None,
            metadata={"model": self.model, "provider": "zhipu"},
        )

        return [result]

    def is_available(self) -> bool:
        """
        Check if the backend is available.

        Returns:
            True if the client is initialized, False otherwise.
        """
        return self.client is not None
