"""
PaddleOCR backend for OCR - High-quality Chinese/English OCR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from notely.ocr.base import OCRBackend, OCRResult, TextBlock


class PaddleOCRBackend(OCRBackend):
    """
    PaddleOCR backend for text recognition.

    PaddleOCR provides:
    - Excellent Chinese/English OCR
    - Table recognition
    - Formula detection (with PP-Structure)
    - Layout analysis

    Args:
        lang: Language code ("ch", "en", "korean", "japan", etc.)
        use_gpu: Whether to use GPU acceleration.
        use_angle_cls: Whether to detect text orientation.
        use_structure: Whether to use structure analysis (tables, formulas).
    """

    def __init__(
        self,
        lang: str = "ch",
        use_gpu: bool = True,
        use_angle_cls: bool = True,
        use_structure: bool = False,
    ):
        self.lang = lang
        self.use_gpu = use_gpu
        self.use_angle_cls = use_angle_cls
        self.use_structure = use_structure
        self._ocr = None
        self._structure = None

    def _load_ocr(self) -> Any:
        """Lazy load PaddleOCR."""
        try:
            from paddleocr import PaddleOCR  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError(
                "PaddleOCR is not installed. Install it with: pip install paddleocr paddlepaddle"
            )

        return PaddleOCR(
            use_angle_cls=self.use_angle_cls,
            lang=self.lang,
            use_gpu=self.use_gpu,
            show_log=False,
        )

    def _load_structure(self) -> Any:
        """Lazy load PP-Structure for table/formula detection."""
        try:
            from paddleocr import PPStructure
        except ImportError:
            return None

        return PPStructure(
            use_gpu=self.use_gpu,
            lang=self.lang,
            show_log=False,
        )

    @property
    def ocr(self) -> Any:
        """Get OCR engine (lazy loading)."""
        if self._ocr is None:
            self._ocr = self._load_ocr()
        return self._ocr

    @property
    def structure(self) -> Any:
        """Get structure engine (lazy loading)."""
        if self._structure is None and self.use_structure:
            self._structure = self._load_structure()
        return self._structure

    def recognize(self, image_path: Path | str) -> OCRResult:
        """
        Recognize text in an image.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult with detected text blocks.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        result = self.ocr.ocr(str(image_path), cls=self.use_angle_cls)

        text_blocks = []
        if result and result[0]:
            for line in result[0]:
                if line:
                    bbox = self._parse_bbox(line[0])
                    text = line[1][0]
                    confidence = line[1][1]

                    # Determine block type based on position and size
                    block_type = self._classify_block(text, bbox)

                    text_blocks.append(
                        TextBlock(
                            text=text,
                            confidence=confidence,
                            bbox=bbox,
                            block_type=block_type,
                        )
                    )

        return OCRResult(
            text_blocks=text_blocks,
            source_path=str(image_path),
            metadata={"backend": "paddleocr", "lang": self.lang},
        )

    def recognize_pdf(self, pdf_path: Path | str) -> list[OCRResult]:
        """
        Recognize text in a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of OCRResult, one per page.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            import fitz  # type: ignore[import-not-found] # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF processing. Install it with: pip install PyMuPDF"
            )

        results = []
        doc = fitz.open(str(pdf_path))

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Render page to image at 150 DPI
            mat = fitz.Matrix(150 / 72, 150 / 72)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image for OCR
            import io

            from PIL import Image

            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))

            # Save temp image for OCR
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img.save(tmp.name)
                ocr_result = self.recognize(tmp.name)
                ocr_result.page_number = page_num + 1
                results.append(ocr_result)

        doc.close()
        return results

    def recognize_table(self, image_path: Path | str) -> str:
        """
        Recognize a table and return as Markdown.

        Args:
            image_path: Path to the image containing a table.

        Returns:
            Markdown table string.
        """
        if not self.use_structure or self.structure is None:
            raise ValueError("Structure analysis is not enabled")

        result = self.structure(str(image_path))

        for region in result:
            if region["type"] == "table":
                return region.get("res", {}).get("html", "")  # type: ignore[no-any-return]

        return ""

    @staticmethod
    def _parse_bbox(coords: list[Any]) -> tuple[int, int, int, int]:
        """Parse bounding box coordinates."""
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    @staticmethod
    def _classify_block(text: str, bbox: tuple[int, int, int, int]) -> str:
        """Classify the type of text block."""
        # Simple heuristics for block classification
        _x1, y1, _x2, _y2 = bbox

        # Short text at top of slide is likely a title
        if len(text) < 50 and y1 < 100:
            return "title"

        # Check for formula patterns
        if any(c in text for c in ["∫", "∑", "√", "∂", "α", "β", "γ", "≈", "≤", "≥"]):
            return "formula"

        return "text"

    def is_available(self) -> bool:
        """Check if PaddleOCR is available."""
        try:
            import paddleocr  # noqa: F401

            return True
        except ImportError:
            return False
