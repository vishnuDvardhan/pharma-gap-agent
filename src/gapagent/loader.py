"""Load requirements, markets, products and dossiers. Dossiers are chunked by heading."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

from .schema import Chunk, Dossier, Product, Requirement

SECTION_RE = re.compile(r"^(\d+(?:\.[0-9A-Za-z]+)*)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
MAX_CHUNK_CHARS = 1500


def section_parts(section: str) -> list[str]:
    return [p.lower() for p in section.split(".") if p]


def section_matches(req_section: str, chunk_section: str) -> bool:
    """Segment-wise prefix match in both directions.

    "3.2.P.8" matches "3.2.P.8.3" (child) and "2.5.6" matches file "2.5" (parent),
    but "1.1" does NOT match "1.14".
    """
    a, b = section_parts(req_section), section_parts(chunk_section)
    n = min(len(a), len(b))
    return a[:n] == b[:n]


# ---------------------------------------------------------------- requirements
def load_markets(data_dir: Path) -> dict:
    return yaml.safe_load((data_dir / "markets.yaml").read_text())["markets"]


def _load_req_file(path: Path) -> list[Requirement]:
    raw = yaml.safe_load(path.read_text())
    market = raw["market"]
    return [Requirement(market=market, **r) for r in raw["requirements"]]


def load_requirements(data_dir: Path, markets: list[str] | None = None) -> dict[str, list[Requirement]]:
    """Return {"ICH": [...], "US": [...], ...}. Raises on duplicate IDs or bad overrides."""
    req_dir = data_dir / "requirements"
    market_cfg = load_markets(data_dir)
    out: dict[str, list[Requirement]] = {"ICH": _load_req_file(req_dir / "ich_core.yaml")}
    for m in markets or list(market_cfg):
        if m not in market_cfg:
            raise ValueError(f"Unknown market {m!r}. Known: {sorted(market_cfg)}")
        out[m] = _load_req_file(req_dir / market_cfg[m]["overlay"])
    ids = [r.id for reqs in out.values() for r in reqs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate requirement IDs: {sorted(dupes)}")
    core_ids = {r.id for r in out["ICH"]}
    for reqs in out.values():
        for r in reqs:
            if r.overrides and r.overrides not in core_ids:
                raise ValueError(f"{r.id} overrides unknown core requirement {r.overrides}")
            if r.check_type == "rule" and r.rule is None:
                raise ValueError(f"{r.id} is a rule check without a rule spec")
    return out


def load_product(data_dir: Path, product_id: str) -> Product:
    return Product(**yaml.safe_load((data_dir / "products" / f"{product_id}.yaml").read_text()))


# ---------------------------------------------------------------- dossiers
def _file_meta(rel: Path) -> tuple[str, str | None, str]:
    parts = rel.parts
    module = parts[0]
    region = parts[1].lower() if module == "m1" and len(parts) > 2 else None
    m = SECTION_RE.match(rel.stem)
    section = m.group(1) if m else module[1:]
    return module, region, section


def _split_long(text: str) -> list[str]:
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    parts, cur = [], ""
    for para in text.split("\n\n"):
        if cur and len(cur) + len(para) > MAX_CHUNK_CHARS:
            parts.append(cur)
            cur = ""
        cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        parts.append(cur)
    return parts


def chunk_markdown(text: str, dossier_id: str, rel: Path) -> list[Chunk]:
    module, region, file_section = _file_meta(rel)
    lines = text.splitlines()
    blocks: list[tuple[str, int, list[str]]] = []   # (heading, start_line, lines)
    heading, start, buf = rel.stem, 1, []
    for i, line in enumerate(lines, start=1):
        hm = HEADING_RE.match(line)
        if hm:
            if any(b.strip() for b in buf):
                blocks.append((heading, start, buf))
            heading, start, buf = hm.group(2).strip(), i, [line]
        else:
            buf.append(line)
    if any(b.strip() for b in buf):
        blocks.append((heading, start, buf))

    chunks: list[Chunk] = []
    for heading, start, buf in blocks:
        section = file_section
        hm = SECTION_RE.match(heading)
        # Accept a heading number only if it is a child of the file's CTD section.
        # (SmPC "6.3 Shelf life" inside file 1.3.1 is not CTD section 6.3.)
        if hm and section_parts(hm.group(1))[: len(section_parts(file_section))] == section_parts(file_section):
            section = hm.group(1)
        body = "\n".join(buf).strip()
        offset = 0
        for piece in _split_long(body):
            n_lines = piece.count("\n") + 1
            chunks.append(Chunk(
                chunk_id=f"{dossier_id}:{rel.stem}#{len(chunks)}",
                dossier_id=dossier_id, file=rel.stem, path=str(rel), module=module,
                region=region, section=section, heading=heading, text=piece,
                line_start=start + offset, line_end=start + offset + n_lines - 1,
            ))
            offset += n_lines
    return chunks


def chunk_pdf(path: Path, dossier_id: str, rel: Path) -> list[Chunk]:  # pragma: no cover - optional dep
    import fitz  # PyMuPDF

    module, region, section = _file_meta(rel)
    chunks = []
    with fitz.open(path) as doc:
        for pno, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                chunks.append(Chunk(
                    chunk_id=f"{dossier_id}:{rel.stem}#p{pno}", dossier_id=dossier_id,
                    file=rel.stem, path=str(rel), module=module, region=region,
                    section=section, heading=f"page {pno}", text=text,
                    line_start=pno, line_end=pno,
                ))
    return chunks


def load_dossier(root: Path) -> Dossier:
    meta = yaml.safe_load((root / "manifest.yaml").read_text())
    chunks: list[Chunk] = []
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.yaml":
            continue
        rel = path.relative_to(root)
        if path.suffix == ".md":
            text = path.read_text(encoding="utf-8")
            files[rel.stem] = text
            chunks.extend(chunk_markdown(text, meta["dossier_id"], rel))
        elif path.suffix == ".pdf":
            pdf_chunks = chunk_pdf(path, meta["dossier_id"], rel)
            files[rel.stem] = "\n".join(c.text for c in pdf_chunks)
            chunks.extend(pdf_chunks)
    return Dossier(
        dossier_id=meta["dossier_id"], title=meta["title"], product=meta["product"],
        reference_market=meta["reference_market"],
        data_classification=meta.get("data_classification", "confidential"),
        root=str(root), chunks=chunks, files=files,
    )


def tree_hash(*paths: Path) -> str:
    """Content hash of every file under the given paths (corpus version for the audit trail)."""
    h = hashlib.sha256()
    for base in paths:
        for p in sorted(base.rglob("*")) if base.is_dir() else [base]:
            if p.is_file():
                h.update(str(p.relative_to(base.parent)).encode())
                h.update(p.read_bytes())
    return h.hexdigest()[:16]
