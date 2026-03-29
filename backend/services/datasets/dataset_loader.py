"""
ARKEN — Dataset Loader v3.0
==============================
v3.0 Changes over v2.0 (PROBLEM 4 FIX):

  Image path resolution completely rewritten:
    - _resolve_image_path() tries multiple path patterns before giving up.
      The actual folder structure is:
        {dataset_root}/{room_type}/{style}/{filename}.jpg
      but the CSV may reference: ../data/raw/{room_type}/{style}/{filename}.jpg
      or just:                   {filename}.jpg
      or:                        {room_type}/{style}/{filename}.jpg
      _resolve_image_path tries all of these systematically.

    - Image existence validation added: per-missing-image warning logged, but
      loading continues for records that do have valid paths.

    - dataset_health_report() added: after loading, prints
      "X of Y image records have valid image paths" for each dataset.

    - load() skips records where the image file does not exist (with a debug
      log), rather than crashing or silently including broken paths.

  All other API (InteriorImageRecord, MaterialStyleDatasetLoader,
  InteriorImagesMetaLoader, DIYRenovationDatasetLoader, ARKENDatasetRegistry)
  UNCHANGED from v2.0.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

DATASET_ROOT = Path(os.getenv("ARKEN_DATASET_DIR", "/app/data/datasets"))

MATERIAL_STYLE_DIR = DATASET_ROOT / "interior_design_material_style"
IMAGES_META_DIR    = DATASET_ROOT / "interior_design_images_metadata"
DIY_DIR            = DATASET_ROOT / "diy_renovation"
INDIA_DIY_DIR      = DATASET_ROOT / "india_diy_knowledge"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
KNOWN_ROOM_FOLDERS = {
    "bathroom", "bedroom", "kitchen", "living_room",
    "dining_room", "study", "hallway",
}


@dataclass
class InteriorImageRecord:
    image_path: Path
    room_type: str = "unknown"
    style: str = "unknown"
    materials: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    source_dataset: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DIYKnowledgeChunk:
    chunk_id: str
    category: str
    video_title: str
    chapter_title: str
    content: str
    clip_link: str
    start_time: float = 0.0
    end_time: float = 0.0

    def to_rag_doc(self) -> Dict[str, Any]:
        return {
            "id": self.chunk_id,
            "text": f"[{self.category}] {self.chapter_title}\n{self.content}",
            "metadata": {
                "category":      self.category,
                "video_title":   self.video_title,
                "chapter_title": self.chapter_title,
                "clip_link":     self.clip_link,
                "source":        "diy_renovation_dataset",
            },
        }


def _infer_room_from_path(p: Path) -> str:
    for part in p.parts:
        part_norm = part.lower().replace(" ", "_").replace("-", "_")
        if part_norm in KNOWN_ROOM_FOLDERS:
            return part_norm
    return "unknown"


def _load_csvs_from_dir(root: Path) -> Dict[str, Dict]:
    result = {}
    for csv_name in ["metadata.csv", "train_data.csv", "val_data.csv", "test_data.csv"]:
        csv_path = root / csv_name
        if not csv_path.exists():
            continue
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    key = (
                        row.get("image_file") or row.get("filename")
                        or row.get("image") or row.get("file") or ""
                    )
                    if key:
                        result[key] = row
                        result[Path(key).stem] = row
        except Exception as e:
            logger.warning(f"[DatasetLoader] CSV error {csv_name}: {e}")
    return result


def _parse_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        if not value.strip():
            return []
        if value.startswith("["):
            try:
                return json.loads(value)
            except Exception:
                pass
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEM 4 FIX: robust image path resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_image_path(
    raw_csv_path: str,
    dataset_root: Path,
    room_type: str,
    style: str,
) -> Optional[Path]:
    """
    Resolve a (potentially broken) CSV image path to a real file on disk.

    Tries multiple path patterns in order:
      1. Absolute path as-is (rare, but possible)
      2. dataset_root / raw_csv_path (relative to dataset root)
      3. dataset_root / room_type / style / filename
      4. dataset_root / room_type / filename
      5. dataset_root / filename
      6. Strip leading "../data/raw/" or "../data/" prefix and retry 3-5

    Returns the first Path that exists, or None if all fail.
    """
    if not raw_csv_path:
        return None

    filename = Path(raw_csv_path).name  # just the filename portion

    # 1. Absolute
    p = Path(raw_csv_path)
    if p.is_absolute() and p.exists():
        return p

    # 2. Relative to dataset_root
    p = dataset_root / raw_csv_path
    if p.exists():
        return p

    # 3. dataset_root / room_type / style / filename
    if room_type and room_type != "unknown" and style and style != "unknown":
        p = dataset_root / room_type / style / filename
        if p.exists():
            return p

    # 4. dataset_root / room_type / filename
    if room_type and room_type != "unknown":
        p = dataset_root / room_type / filename
        if p.exists():
            return p

    # 5. dataset_root / filename
    p = dataset_root / filename
    if p.exists():
        return p

    # 6. Strip known broken prefixes and retry 3-5
    stripped = raw_csv_path
    for prefix in ("../data/raw/", "../data/", "data/raw/", "data/"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break

    if stripped != raw_csv_path:
        stem_path = Path(stripped)
        for candidate in [
            dataset_root / stripped,
            dataset_root / room_type / style / stem_path.name if (room_type and style) else None,
            dataset_root / room_type / stem_path.name if room_type else None,
            dataset_root / stem_path.name,
        ]:
            if candidate is not None and candidate.exists():
                return candidate

    return None


def dataset_health_report(
    source_name: str,
    total_csv_records: int,
    valid_records: int,
    invalid_records: int,
) -> None:
    """
    Log a structured health report for a dataset after loading.
    Called after load() completes in _BaseImageLoader.
    """
    pct = valid_records / max(total_csv_records, 1) * 100
    logger.info(
        f"[DatasetHealth] {source_name}: "
        f"{valid_records} of {total_csv_records} image records have valid image paths "
        f"({pct:.1f}% valid | {invalid_records} skipped)"
    )
    if valid_records == 0:
        logger.error(
            f"[DatasetHealth] {source_name}: NO valid images found. "
            "Check that image directories exist and CSV paths match the folder structure. "
            "Expected: {dataset_root}/{room_type}/{style}/{filename}.jpg"
        )
    elif pct < 50:
        logger.warning(
            f"[DatasetHealth] {source_name}: fewer than 50% of records have valid paths. "
            "Review path resolution — most images may be in a different subfolder layout."
        )


class _BaseImageLoader:
    """
    Shared loading + indexing logic for both image datasets.

    v3.0: _resolve_image_path() replaces direct path access; records with
    unresolvable paths are skipped with a debug log, not crashed on.
    dataset_health_report() is called after every load().
    """

    def __init__(self, root: Path, source_name: str):
        self.root = root
        self.source_name = source_name
        self._records: Optional[List[InteriorImageRecord]] = None
        self._style_index: Dict[str, List[InteriorImageRecord]] = {}
        self._room_index:  Dict[str, List[InteriorImageRecord]] = {}
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.root.exists() and any(self.root.rglob("*" + ext) for ext in IMAGE_EXTS)

    def load(self) -> List[InteriorImageRecord]:
        with self._lock:
            if self._records is not None:
                return self._records

            if not self.root.exists():
                logger.warning(f"[{self.source_name}] Not found: {self.root}")
                self._records = []
                return self._records

            meta_map = _load_csvs_from_dir(self.root)
            records  = []
            skipped  = 0
            # Count how many CSV entries we attempted to resolve
            csv_entry_count = 0

            # ── PROBLEM 4 FIX: load from CSV records using path resolution ────
            # Walk all entries in the CSV maps and resolve each image path.
            # Fallback: also walk filesystem for images not in the CSV.
            seen_paths: set = set()

            for img_path in sorted(self.root.rglob("*")):
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue

                # Image is physically present; infer metadata from CSV if available
                folder_room = _infer_room_from_path(img_path)
                meta        = meta_map.get(img_path.name, meta_map.get(img_path.stem, {}))

                style = (
                    meta.get("style") or meta.get("design_style")
                    or meta.get("label") or _infer_style_from_path(img_path) or "unknown"
                )

                seen_paths.add(img_path)
                records.append(InteriorImageRecord(
                    image_path=img_path,
                    room_type=meta.get("room_type", folder_room),
                    style=style,
                    materials=_parse_list(meta.get("materials", meta.get("material", ""))),
                    objects=_parse_list(meta.get("objects", meta.get("furniture", ""))),
                    source_dataset=self.source_name,
                    metadata=meta,
                ))

            # ── Also resolve any CSV entries that point to images NOT found by rglob ──
            for raw_key, meta in meta_map.items():
                # raw_key might be a path from the CSV (e.g. "../data/raw/kitchen/modern/x.jpg")
                raw_image_ref = (
                    meta.get("image_file") or meta.get("filename")
                    or meta.get("image") or meta.get("file") or ""
                )
                if not raw_image_ref:
                    continue

                csv_entry_count += 1

                # Try to infer room/style from meta or path
                folder_room = (
                    meta.get("room_type")
                    or _infer_room_from_path(Path(raw_image_ref))
                    or "unknown"
                )
                style = (
                    meta.get("style") or meta.get("design_style")
                    or meta.get("label")
                    or _infer_style_from_path(Path(raw_image_ref))
                    or "unknown"
                )

                resolved = _resolve_image_path(
                    raw_csv_path=raw_image_ref,
                    dataset_root=self.root,
                    room_type=folder_room,
                    style=style,
                )

                if resolved is None:
                    skipped += 1
                    logger.debug(
                        f"[{self.source_name}] Could not resolve image path: '{raw_image_ref}' "
                        f"(room={folder_room}, style={style})"
                    )
                    continue

                if resolved in seen_paths:
                    # Already captured by the rglob walk above
                    continue

                seen_paths.add(resolved)
                records.append(InteriorImageRecord(
                    image_path=resolved,
                    room_type=folder_room,
                    style=style,
                    materials=_parse_list(meta.get("materials", meta.get("material", ""))),
                    objects=_parse_list(meta.get("objects", meta.get("furniture", ""))),
                    source_dataset=self.source_name,
                    metadata=meta,
                ))

            self._records = records
            self._build_indexes()

            # ── PROBLEM 4 FIX: health report ─────────────────────────────────
            total_attempted = max(csv_entry_count, len(records) + skipped)
            dataset_health_report(
                source_name=self.source_name,
                total_csv_records=total_attempted,
                valid_records=len(records),
                invalid_records=skipped,
            )
            logger.info(f"[{self.source_name}] Loaded {len(records)} images from {self.root}")
            return self._records

    def _build_indexes(self):
        self._style_index = {}
        self._room_index  = {}
        for rec in (self._records or []):
            self._style_index.setdefault(rec.style.lower().strip(), []).append(rec)
            self._room_index.setdefault(rec.room_type.lower().strip(), []).append(rec)

    def get_by_style(self, style: str) -> List[InteriorImageRecord]:
        self.load()
        key     = style.lower().strip()
        results = list(self._style_index.get(key, []))
        if not results:
            for idx_key, recs in self._style_index.items():
                if key in idx_key or idx_key in key:
                    results.extend(recs)
        return results[:10]

    def get_by_room_type(self, room_type: str) -> List[InteriorImageRecord]:
        self.load()
        return self._room_index.get(room_type.lower().strip(), [])[:10]

    def iter_batches(self, batch_size: int = 32) -> Generator[List[InteriorImageRecord], None, None]:
        records = self.load()
        for i in range(0, len(records), batch_size):
            yield records[i:i + batch_size]


def _infer_style_from_path(p: Path) -> Optional[str]:
    """
    Guess design style from a path component.
    e.g. …/living_room/scandinavian/living_room_scandinavian_5.jpg → "scandinavian"
    """
    known_styles = {
        "modern", "scandinavian", "minimalist", "industrial",
        "bohemian", "traditional", "contemporary", "japandi",
        "art_deco", "rustic", "coastal",
    }
    for part in p.parts:
        part_norm = part.lower().replace(" ", "_").replace("-", "_")
        if part_norm in known_styles:
            return part_norm
    # Also try the filename stem: living_room_scandinavian_5 → "scandinavian"
    stem_parts = p.stem.lower().replace("-", "_").split("_")
    for sp in stem_parts:
        if sp in known_styles:
            return sp
    return None


class MaterialStyleDatasetLoader(_BaseImageLoader):
    def __init__(self, root: Path = MATERIAL_STYLE_DIR):
        super().__init__(root, "interior_design_material_style")


class InteriorImagesMetaLoader(_BaseImageLoader):
    def __init__(self, root: Path = IMAGES_META_DIR):
        super().__init__(root, "interior_design_images_metadata")


class DIYRenovationDatasetLoader:
    def __init__(self, root: Optional[Path] = None):
        # Prefer India DIY knowledge base over old US YouTube transcript dataset
        if root is None:
            india_csv = INDIA_DIY_DIR / "india_diy_knowledge.csv"
            if india_csv.exists():
                root = INDIA_DIY_DIR
            else:
                root = DIY_DIR
        self.root = root
        self._chunks: Optional[List[DIYKnowledgeChunk]] = None
        self._category_index: Dict[str, List[DIYKnowledgeChunk]] = {}
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self._find_csv())

    def _find_csv(self) -> Optional[Path]:
        # Check India DIY knowledge first (preferred — real India-specific content)
        india_csv = INDIA_DIY_DIR / "india_diy_knowledge.csv"
        if india_csv.exists():
            return india_csv

        # Fall back to legacy US YouTube transcript dataset
        for name in ["DIY_dataset.csv", "diy_dataset.csv", "dataset.csv"]:
            p = self.root / name
            if p.exists():
                logger.warning(
                    f"[DIYLoader] Using legacy US YouTube DIY dataset at {p}. "
                    "Run build_india_diy_dataset.py to generate India-specific content."
                )
                return p
        for p in self.root.glob("*.csv"):
            return p
        return None

    def load(self) -> List[DIYKnowledgeChunk]:
        with self._lock:
            if self._chunks is not None:
                return self._chunks

            csv_path = self._find_csv()
            if not csv_path:
                logger.warning(f"[DIYLoader] DIY_dataset.csv not found in {self.root}")
                self._chunks = []
                return self._chunks

            chunks = []
            try:
                with open(csv_path, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        if not row.get("content", "").strip():
                            continue
                        chunks.append(DIYKnowledgeChunk(
                            chunk_id=row.get("chapter_id", f"diy_{len(chunks)}"),
                            category=row.get("playlist_title", "General"),
                            video_title=row.get("video_title", ""),
                            chapter_title=row.get("chapter_title", ""),
                            content=row.get("content", "").strip(),
                            clip_link=row.get("clip_link", ""),
                            start_time=_safe_float(row.get("start_time", "0")),
                            end_time=_safe_float(row.get("end_time", "0")),
                        ))
            except Exception as e:
                logger.error(f"[DIYLoader] Failed to load: {e}", exc_info=True)
                self._chunks = []
                return self._chunks

            self._chunks = chunks
            for chunk in chunks:
                self._category_index.setdefault(chunk.category.lower(), []).append(chunk)

            logger.info(
                f"[DIYLoader] Loaded {len(chunks)} chunks | "
                f"categories: {list(self._category_index.keys())}"
            )
            return self._chunks

    def get_by_category(self, category: str) -> List[DIYKnowledgeChunk]:
        self.load()
        return self._category_index.get(category.lower(), [])

    def get_relevant_chunks(
        self, query_terms: List[str], max_results: int = 10,
    ) -> List[DIYKnowledgeChunk]:
        lower_terms = [t.lower() for t in query_terms]
        scored = []
        for chunk in self.load():
            text  = (chunk.content + " " + chunk.chapter_title + " " + chunk.category).lower()
            score = sum(1 for t in lower_terms if t in text)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:max_results]]

    def to_rag_documents(self) -> List[Dict[str, Any]]:
        return [c.to_rag_doc() for c in self.load()]

    def get_categories(self) -> List[str]:
        return sorted(self._category_index.keys())


class ARKENDatasetRegistry:
    _instance: Optional["ARKENDatasetRegistry"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.material_style  = MaterialStyleDatasetLoader()
        self.interior_images = InteriorImagesMetaLoader()
        self.diy_renovation  = DIYRenovationDatasetLoader()
        self._rag_ingested   = False
        # Corpus-backed handles (Task 4)
        self._diy_handle     = None
        self._design_handle  = None

    @property
    def diy_handle(self) -> "DIYDatasetHandle":
        if self._diy_handle is None:
            self._diy_handle = DIYDatasetHandle(self.diy_renovation)
        return self._diy_handle

    @property
    def interior_design(self) -> "InteriorDesignHandle":
        if self._design_handle is None:
            self._design_handle = InteriorDesignHandle(self.material_style)
        return self._design_handle

    @classmethod
    def get(cls) -> "ARKENDatasetRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def status(self) -> Dict[str, Any]:
        return {
            "material_style_dataset": {
                "available": self.material_style.available,
                "path":      str(MATERIAL_STYLE_DIR),
            },
            "interior_images_dataset": {
                "available": self.interior_images.available,
                "path":      str(IMAGES_META_DIR),
            },
            "diy_renovation_dataset": {
                "available": self.diy_renovation.available,
                "path":      str(DIY_DIR),
                "chunks":    len(self.diy_renovation.load()) if self.diy_renovation.available else 0,
            },
            "rag_ingested": self._rag_ingested,
        }

    def startup_ingest(self):
        if self._rag_ingested:
            return
        if not self.diy_renovation.available:
            logger.warning("[DatasetRegistry] DIY dataset not available — skipping RAG ingestion")
            return

        try:
            docs = self.diy_renovation.to_rag_documents()
            logger.info(f"[DatasetRegistry] Ingesting {len(docs)} DIY chunks into RAG...")

            try:
                from services.rag.knowledge_loader import KnowledgeLoader
                loader = KnowledgeLoader()
                if hasattr(loader, "ingest_documents"):
                    loader.ingest_documents(docs)
                    self._rag_ingested = True
                    logger.info(f"[DatasetRegistry] ✓ {len(docs)} DIY chunks ingested via KnowledgeLoader")
                    return
            except Exception as e:
                logger.debug(f"[DatasetRegistry] KnowledgeLoader path: {e}")

            try:
                from services.rag.vector_store import VectorStore
                vs = VectorStore()
                if hasattr(vs, "add_documents"):
                    vs.add_documents(
                        texts=[d["text"] for d in docs],
                        metadatas=[d["metadata"] for d in docs],
                        ids=[d["id"] for d in docs],
                    )
                    self._rag_ingested = True
                    logger.info(f"[DatasetRegistry] ✓ {len(docs)} DIY chunks ingested via VectorStore")
            except Exception as e:
                logger.warning(f"[DatasetRegistry] VectorStore ingestion failed: {e}")

        except Exception as e:
            logger.error(f"[DatasetRegistry] startup_ingest error: {e}", exc_info=True)

# ─────────────────────────────────────────────────────────────────────────────
# CORPUS-BACKED HANDLES — DatasetHandle wrappers over ChromaDB corpus
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass


@dataclass
class Chunk:
    """Knowledge chunk returned by DatasetHandle queries."""
    chunk_id: str
    chapter_title: str
    category: str
    content: str
    clip_link: str
    source_dataset: str


@dataclass
class DesignRecord:
    """Design record returned by InteriorDesignHandle.search()."""
    record_id: str
    style: str
    room_type: str
    content: str
    source_dataset: str
    metadata: Dict[str, Any]


class DIYDatasetHandle:
    """
    DatasetHandle for diy_renovation domain backed by ChromaDB corpus.
    Falls back to DIYRenovationDatasetLoader if corpus unavailable.
    """

    def __init__(self, diy_loader: "DIYRenovationDatasetLoader"):
        self._loader = diy_loader
        self._retriever = None

    def _get_retriever(self):
        if self._retriever is None:
            try:
                from data.rag_knowledge_base.corpus_builder import get_retriever
                self._retriever = get_retriever()
            except Exception as e:
                logger.debug(f"[DIYHandle] Corpus retriever unavailable: {e}")
        return self._retriever

    @property
    def available(self) -> bool:
        retriever = self._get_retriever()
        if retriever and retriever.collection:
            try:
                return retriever.collection.count() > 0
            except Exception:
                pass
        return self._loader.available

    def get_by_category(self, category: str) -> List[Chunk]:
        """Return chunks matching a category from diy_contractor domain."""
        retriever = self._get_retriever()
        if retriever:
            try:
                raw = retriever.retrieve(
                    query=f"DIY renovation {category} India guide",
                    top_k=10,
                    domain_filter="diy_contractor",
                )
                return [
                    Chunk(
                        chunk_id=r["id"],
                        chapter_title=r["metadata"].get("title", ""),
                        category=r["metadata"].get("subcategory", category),
                        content=r["content"],
                        clip_link="",
                        source_dataset="corpus_builder",
                    )
                    for r in raw
                ]
            except Exception as e:
                logger.debug(f"[DIYHandle] Corpus search failed: {e}")

        # Fallback to file-based loader
        chunks_raw = self._loader.get_by_category(category)
        return [
            Chunk(
                chunk_id=c.chunk_id,
                chapter_title=c.chapter_title,
                category=c.category,
                content=c.content,
                clip_link=c.clip_link,
                source_dataset="diy_renovation_csv",
            )
            for c in chunks_raw
        ]

    def search(self, query: str) -> List[Chunk]:
        """Free-text search across diy_contractor corpus."""
        retriever = self._get_retriever()
        if retriever:
            try:
                raw = retriever.retrieve(
                    query=query,
                    top_k=8,
                    domain_filter="diy_contractor",
                )
                return [
                    Chunk(
                        chunk_id=r["id"],
                        chapter_title=r["metadata"].get("title", ""),
                        category=r["metadata"].get("subcategory", ""),
                        content=r["content"],
                        clip_link="",
                        source_dataset="corpus_builder",
                    )
                    for r in raw
                ]
            except Exception as e:
                logger.debug(f"[DIYHandle] Corpus search failed: {e}")

        # Fallback
        terms = query.split()
        chunks_raw = self._loader.get_relevant_chunks(terms, max_results=8)
        return [
            Chunk(
                chunk_id=c.chunk_id,
                chapter_title=c.chapter_title,
                category=c.category,
                content=c.content,
                clip_link=c.clip_link,
                source_dataset="diy_renovation_csv",
            )
            for c in chunks_raw
        ]


class InteriorDesignHandle:
    """
    DatasetHandle for interior_design domain backed by ChromaDB corpus.
    Falls back to MaterialStyleDatasetLoader if corpus unavailable.
    """

    def __init__(self, style_loader: "MaterialStyleDatasetLoader"):
        self._loader = style_loader
        self._retriever = None

    def _get_retriever(self):
        if self._retriever is None:
            try:
                from data.rag_knowledge_base.corpus_builder import get_retriever
                self._retriever = get_retriever()
            except Exception as e:
                logger.debug(f"[DesignHandle] Corpus retriever unavailable: {e}")
        return self._retriever

    @property
    def available(self) -> bool:
        retriever = self._get_retriever()
        if retriever and retriever.collection:
            try:
                return retriever.collection.count() > 0
            except Exception:
                pass
        return self._loader.available

    def search(self, style: str = "", room_type: str = "") -> List[DesignRecord]:
        """Search design corpus by style and/or room type."""
        retriever = self._get_retriever()
        query_parts = [p for p in [style, room_type, "interior design India"] if p]
        query = " ".join(query_parts)

        if retriever:
            try:
                raw = retriever.retrieve(
                    query=query,
                    top_k=8,
                    domain_filter="design_styles",
                )
                return [
                    DesignRecord(
                        record_id=r["id"],
                        style=r["metadata"].get("style_relevance", style),
                        room_type=r["metadata"].get("room_relevance", room_type),
                        content=r["content"],
                        source_dataset="corpus_builder",
                        metadata=r["metadata"],
                    )
                    for r in raw
                ]
            except Exception as e:
                logger.debug(f"[DesignHandle] Corpus search failed: {e}")

        # Fallback to image loader
        records: List[InteriorImageRecord] = []
        if style:
            records.extend(self._loader.get_by_style(style))
        if room_type:
            records.extend(self._loader.get_by_room_type(room_type))

        seen: set = set()
        design_records = []
        for rec in records:
            if str(rec.image_path) not in seen:
                seen.add(str(rec.image_path))
                design_records.append(DesignRecord(
                    record_id=str(rec.image_path),
                    style=rec.style,
                    room_type=rec.room_type,
                    content=f"Style: {rec.style}. Room: {rec.room_type}. Materials: {', '.join(rec.materials)}",
                    source_dataset=rec.source_dataset,
                    metadata=rec.metadata,
                ))
        return design_records[:8]