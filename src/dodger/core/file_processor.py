"""File processor for handling document and image uploads.

This module provides:
- Document parsing (PDF, DOCX, TXT, MD)
- Image processing for vision models
- Code file handling with syntax detection
- File chunking for large documents
"""

from __future__ import annotations

import base64
import io
import mimetypes
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4

import structlog

from dodger.config import get_data_dir
from dodger.core.exceptions import FileProcessingError

# Lazy imports for optional dependencies
DOCX_AVAILABLE = False
PYPDF_AVAILABLE = False

try:
    from pypdf import PdfReader

    PYPDF_AVAILABLE = True
except ImportError:
    PdfReader = None  # type: ignore[misc, assignment]

try:
    from docx import Document as DocxDocument

    DOCX_AVAILABLE = True
except ImportError:
    DocxDocument = None  # type: ignore[misc, assignment]

logger = structlog.get_logger(__name__)


class FileType(str, Enum):
    """Supported file types."""

    PDF = "pdf"
    DOCX = "docx"
    TEXT = "text"
    MARKDOWN = "markdown"
    CODE = "code"
    IMAGE = "image"
    UNKNOWN = "unknown"


@dataclass
class ProcessedFile:
    """Result of processing a file.

    Attributes:
        id: Unique file identifier.
        filename: Original filename.
        file_type: Detected file type.
        mime_type: MIME type of the file.
        size_bytes: File size in bytes.
        extracted_text: Text extracted from the file.
        chunks: Text chunks for large documents.
        image_data: Base64 encoded image for vision models.
        stored_path: Path where the file is stored.
        metadata: Additional file metadata.
    """

    id: UUID = field(default_factory=uuid4)
    filename: str = ""
    file_type: FileType = FileType.UNKNOWN
    mime_type: str = ""
    size_bytes: int = 0
    extracted_text: str = ""
    chunks: list[str] = field(default_factory=list)
    image_data: str | None = None
    stored_path: Path | None = None
    metadata: dict = field(default_factory=dict)


# MIME type to file type mapping
MIME_TYPE_MAP: dict[str, FileType] = {
    "application/pdf": FileType.PDF,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": FileType.DOCX,
    "application/msword": FileType.DOCX,
    "text/plain": FileType.TEXT,
    "text/markdown": FileType.MARKDOWN,
    "text/x-markdown": FileType.MARKDOWN,
    "image/png": FileType.IMAGE,
    "image/jpeg": FileType.IMAGE,
    "image/gif": FileType.IMAGE,
    "image/webp": FileType.IMAGE,
}

# Code file extensions
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".r",
    ".sql",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".yaml",
    ".yml",
    ".json",
    ".xml",
    ".html",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".vue",
    ".svelte",
}


class FileProcessor:
    """Processes uploaded files for use with LLMs.

    Handles:
    - Document text extraction
    - Image encoding for vision models
    - Code file syntax detection
    - Large file chunking
    """

    def __init__(self, storage_dir: Path | None = None) -> None:
        """Initialize the file processor.

        Args:
            storage_dir: Directory for storing processed files.
        """
        self.storage_dir = storage_dir or get_data_dir() / "uploads"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Chunk settings
        self.chunk_size = 4000  # characters
        self.chunk_overlap = 200

    def detect_file_type(self, filename: str, mime_type: str | None = None) -> FileType:
        """Detect the type of a file.

        Args:
            filename: The filename.
            mime_type: Optional MIME type hint.

        Returns:
            The detected file type.
        """
        # Try MIME type first
        if mime_type and mime_type in MIME_TYPE_MAP:
            return MIME_TYPE_MAP[mime_type]

        # Check by extension
        ext = Path(filename).suffix.lower()

        if ext in CODE_EXTENSIONS:
            return FileType.CODE

        if ext == ".pdf":
            return FileType.PDF
        if ext in {".docx", ".doc"}:
            return FileType.DOCX
        if ext == ".txt":
            return FileType.TEXT
        if ext in {".md", ".markdown"}:
            return FileType.MARKDOWN
        if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            return FileType.IMAGE

        # Guess MIME type
        guessed_mime, _ = mimetypes.guess_type(filename)
        if guessed_mime and guessed_mime in MIME_TYPE_MAP:
            return MIME_TYPE_MAP[guessed_mime]

        return FileType.UNKNOWN

    async def process_file(
        self,
        file_path: Path | None = None,
        file_data: bytes | None = None,
        filename: str = "unknown",
        mime_type: str | None = None,
    ) -> ProcessedFile:
        """Process a file for use with LLMs.

        Args:
            file_path: Path to the file (mutually exclusive with file_data).
            file_data: Raw file bytes (mutually exclusive with file_path).
            filename: Original filename.
            mime_type: Optional MIME type hint.

        Returns:
            Processed file with extracted content.
        """
        if file_path is None and file_data is None:
            msg = "Either file_path or file_data must be provided"
            raise FileProcessingError(msg)

        # Read file if path provided
        if file_path:
            if not file_path.exists():
                msg = f"File not found: {file_path}"
                raise FileProcessingError(msg, str(file_path))
            file_data = file_path.read_bytes()
            filename = file_path.name

        if file_data is None:
            msg = "No file data available"
            raise FileProcessingError(msg)

        # Detect type
        file_type = self.detect_file_type(filename, mime_type)
        actual_mime = mime_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"

        result = ProcessedFile(
            filename=filename,
            file_type=file_type,
            mime_type=actual_mime,
            size_bytes=len(file_data),
        )

        # Process based on type
        try:
            if file_type == FileType.PDF:
                result.extracted_text = await self._extract_pdf(file_data)
            elif file_type == FileType.DOCX:
                result.extracted_text = await self._extract_docx(file_data)
            elif file_type in {FileType.TEXT, FileType.MARKDOWN, FileType.CODE}:
                result.extracted_text = self._extract_text(file_data)
            elif file_type == FileType.IMAGE:
                result.image_data = self._encode_image(file_data, actual_mime)
            else:
                # Try as text
                try:
                    result.extracted_text = file_data.decode("utf-8")
                except UnicodeDecodeError as err:
                    msg = f"Unsupported file type: {filename}"
                    raise FileProcessingError(msg, filename) from err

            # Chunk large text
            if result.extracted_text and len(result.extracted_text) > self.chunk_size:
                result.chunks = self._chunk_text(result.extracted_text)

            # Store file
            result.stored_path = await self._store_file(result.id, filename, file_data)

            logger.info(
                "Processed file",
                filename=filename,
                type=file_type.value,
                size=result.size_bytes,
                text_length=len(result.extracted_text),
                chunks=len(result.chunks),
            )

        except FileProcessingError:
            raise
        except Exception as e:
            msg = f"Failed to process file: {e}"
            raise FileProcessingError(
                msg,
                filename,
            ) from e

        return result

    async def _extract_pdf(self, data: bytes) -> str:
        """Extract text from a PDF file."""
        if not PYPDF_AVAILABLE or PdfReader is None:
            msg = "pypdf not installed for PDF processing"
            raise FileProcessingError(msg)

        try:
            reader = PdfReader(io.BytesIO(data))
            text_parts = [
                text
                for page in reader.pages
                if (text := page.extract_text())
            ]
            return "\n\n".join(text_parts)

        except Exception as e:
            msg = f"Failed to extract PDF text: {e}"
            raise FileProcessingError(msg) from e

    async def _extract_docx(self, data: bytes) -> str:
        """Extract text from a DOCX file."""
        if not DOCX_AVAILABLE or DocxDocument is None:
            msg = "python-docx not installed for DOCX processing"
            raise FileProcessingError(msg)

        try:
            doc = DocxDocument(io.BytesIO(data))
            text_parts = [
                para.text
                for para in doc.paragraphs
                if para.text.strip()
            ]
            return "\n\n".join(text_parts)

        except Exception as e:
            msg = f"Failed to extract DOCX text: {e}"
            raise FileProcessingError(msg) from e

    def _extract_text(self, data: bytes) -> str:
        """Extract text from a text file."""
        encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]
        for encoding in encodings:
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue

        msg = "Could not decode text file"
        raise FileProcessingError(msg)

    def _encode_image(self, data: bytes, mime_type: str) -> str:
        """Encode an image as base64 for vision models."""
        base64_data = base64.b64encode(data).decode("utf-8")
        return f"data:{mime_type};base64,{base64_data}"

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks.

        Args:
            text: The text to chunk.

        Returns:
            List of text chunks.
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at a sentence or paragraph
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind("\n\n", start, end)
                if para_break > start + self.chunk_size // 2:
                    end = para_break + 2
                else:
                    # Look for sentence break
                    for punct in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                        sent_break = text.rfind(punct, start, end)
                        if sent_break > start + self.chunk_size // 2:
                            end = sent_break + len(punct)
                            break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap

        return chunks

    async def _store_file(
        self, file_id: UUID, filename: str, data: bytes
    ) -> Path:
        """Store a file to disk.

        Args:
            file_id: Unique file ID.
            filename: Original filename.
            data: File data.

        Returns:
            Path to stored file.
        """
        # Create subdirectory for this file
        file_dir = self.storage_dir / str(file_id)
        file_dir.mkdir(parents=True, exist_ok=True)

        # Store with original extension
        ext = Path(filename).suffix
        stored_path = file_dir / f"file{ext}"
        stored_path.write_bytes(data)

        return stored_path

    def get_context_for_message(
        self,
        processed_file: ProcessedFile,
        max_chars: int = 8000,
    ) -> str:
        """Get file content formatted for inclusion in a message.

        Args:
            processed_file: The processed file.
            max_chars: Maximum characters to include.

        Returns:
            Formatted content string.
        """
        if processed_file.file_type == FileType.IMAGE:
            return f"[Image: {processed_file.filename}]"

        text = processed_file.extracted_text
        if len(text) > max_chars:
            remaining = len(processed_file.extracted_text) - max_chars
            text = f"{text[:max_chars]}\n\n[... truncated, {remaining} more characters ...]"

        file_type = processed_file.file_type.value
        if processed_file.file_type == FileType.CODE:
            # Detect language from extension
            ext = Path(processed_file.filename).suffix.lstrip(".")
            return f"```{ext}\n{text}\n```"

        return f"**File: {processed_file.filename}** ({file_type})\n\n{text}"


# Global file processor instance
file_processor = FileProcessor()
