"""Multi-format document loaders.

Each loader normalizes a file into an ordered list of `Block`s carrying the
provenance the citation layer needs: the nearest section heading and (for PDFs)
the page number. Chunkers consume blocks, so section/page metadata flows all the
way to citations without re-parsing.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Block:
    text: str
    section: str | None = None
    page: int | None = None


@dataclass
class LoadedDocument:
    doc_id: str
    source: str
    blocks: List[Block] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf"}


def doc_id_for(source: str, content: str) -> str:
    h = hashlib.sha256()
    h.update(os.path.basename(source).encode("utf-8"))
    h.update(b"\x00")
    h.update(content.encode("utf-8"))
    return h.hexdigest()[:16]


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_document(path: str) -> LoadedDocument:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".md", ".markdown"):
        return _load_markdown(path)
    if ext in (".html", ".htm"):
        return _load_html(path)
    if ext == ".pdf":
        return _load_pdf(path)
    return _load_text(path)


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


def _load_text(path: str) -> LoadedDocument:
    raw = _read_text(path)
    content = _clean(raw)
    blocks = [Block(text=para) for para in _split_paragraphs(content)]
    return LoadedDocument(doc_id=doc_id_for(path, raw), source=os.path.basename(path), blocks=blocks)


def _load_markdown(path: str) -> LoadedDocument:
    raw = _read_text(path)
    blocks: List[Block] = []
    current_section: str | None = None
    buffer: List[str] = []

    def flush():
        if buffer:
            text = _clean("\n".join(buffer))
            if text:
                blocks.append(Block(text=text, section=current_section))
            buffer.clear()

    for line in raw.split("\n"):
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            flush()
            current_section = heading.group(2).strip()
        else:
            buffer.append(line)
    flush()
    return LoadedDocument(doc_id=doc_id_for(path, raw), source=os.path.basename(path), blocks=blocks)


def _load_html(path: str) -> LoadedDocument:
    from bs4 import BeautifulSoup

    raw = _read_text(path)
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    blocks: List[Block] = []
    current_section: str | None = None
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre"]):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name.startswith("h"):
            current_section = text
        else:
            blocks.append(Block(text=_clean(text), section=current_section))
    if not blocks:  # fallback: whole-page text
        blocks = [Block(text=_clean(soup.get_text(" ", strip=True)))]
    return LoadedDocument(doc_id=doc_id_for(path, raw), source=os.path.basename(path), blocks=blocks)


def _load_pdf(path: str) -> LoadedDocument:
    from pypdf import PdfReader

    reader = PdfReader(path)
    blocks: List[Block] = []
    hashable = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        hashable.append(text)
        for para in _split_paragraphs(text):
            blocks.append(Block(text=para, page=page_num))
    content = "\n".join(hashable)
    return LoadedDocument(doc_id=doc_id_for(path, content), source=os.path.basename(path), blocks=blocks)


def _split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
