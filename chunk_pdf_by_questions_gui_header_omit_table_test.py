"""
Annotate question regions in a PDF using metadata from a JSON file.

Requires PyMuPDF (install with: pip install pymupdf)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import unicodedata
import atexit
import gc
import shutil
from contextlib import contextmanager, nullcontext
from difflib import SequenceMatcher
from bisect import bisect_right
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import fitz  # PyMuPDF
except ImportError as exc:
    raise SystemExit(
        "PyMuPDF is required. Install it via 'pip install pymupdf'."
    ) from exc

try:
    from pypdf import PdfReader, PdfWriter  # type: ignore
except ImportError:
    PdfReader = PdfWriter = None  # type: ignore

try:
    import pdfplumber  # type: ignore
except ImportError:
    pdfplumber = None

try:
    import camelot  # type: ignore
except ImportError:
    camelot = None

try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None

PDFPLUMBER_AVAILABLE = pdfplumber is not None and pd is not None
CAMEL0T_AVAILABLE = PDFPLUMBER_AVAILABLE and camelot is not None and PdfReader is not None

SOFT_HYPHEN = "\u00ad"


def normalize_text(text: str) -> str:
    """Lowercase and strip whitespace / non-letter-or-number chars for matching."""
    text = unicodedata.normalize("NFKC", text.replace(SOFT_HYPHEN, ""))
    # Remove markers like <보기> or <보기 3> that often prefix Korean exam questions.
    if text.startswith("<보기"):
        closing = text.find(">")
        if closing != -1:
            text = text[closing + 1 :]
    keep: List[str] = []
    for ch in text:
        if ch.isspace():
            continue
        category = unicodedata.category(ch)
        if category and category[0] in ("L", "N"):
            keep.append(ch.lower())
    return "".join(keep)


@dataclass
class Question:
    number: int
    text: str
    normalized: str
    raw_entry: dict


@dataclass
class Block:
    page_index: int
    bbox: fitz.Rect
    text: str
    normalized: str
    start: int
    end: int


@dataclass
class OmitRegion:
    page_index: int
    rect: fitz.Rect


class LinearPdfIndex:
    """Flatten a PDF into ordered text blocks with normalized lookup strings."""

    def __init__(
        self,
        doc: fitz.Document,
        *,
        omit_regions: Optional[Sequence[OmitRegion]] = None,
    ) -> None:
        self.blocks: List[Block] = []
        self._global_norm_parts: List[str] = []
        self._starts: List[int] = []
        self.global_normalized: str = ""
        self._omit_by_page: Dict[int, List[fitz.Rect]] = defaultdict(list)
        if omit_regions:
            for region in omit_regions:
                self._omit_by_page[region.page_index].append(region.rect)
        self._linearize(doc)

    def _linearize(self, doc: fitz.Document) -> None:
        cursor = 0
        for page_index in range(doc.page_count):
            page = doc[page_index]
            raw_blocks = page.get_text("blocks") or []
            # Sort: top-to-bottom (y), then left-to-right (x)
            sorted_blocks = sorted(
                raw_blocks, key=lambda b: (round(b[1], 3), round(b[0], 3))
            )
            for raw in sorted_blocks:
                if len(raw) < 5:
                    continue
                x0, y0, x1, y1, raw_text = raw[:5]
                block_type = raw[6] if len(raw) > 6 else 0
                if block_type != 0:
                    continue  # skip images etc.
                rect = fitz.Rect(x0, y0, x1, y1)
                omit_regions = self._omit_by_page.get(page_index)
                if omit_regions and any(rect.intersects(omit) for omit in omit_regions):
                    logging.debug(
                        "Skipping block on page %s due to omit region overlap",
                        page_index + 1,
                    )
                    continue
                cleaned = "\n".join(
                    line.rstrip() for line in (raw_text or "").splitlines()
                ).strip()
                if not cleaned:
                    continue
                normalized = normalize_text(cleaned)
                if not normalized:
                    continue
                start = cursor
                cursor += len(normalized)
                block = Block(
                    page_index=page_index,
                    bbox=rect,
                    text=cleaned,
                    normalized=normalized,
                    start=start,
                    end=cursor,
                )
                self.blocks.append(block)
                self._global_norm_parts.append(normalized)
                self._starts.append(start)
        self.global_normalized = "".join(self._global_norm_parts)
        logging.debug("Indexed %s text blocks", len(self.blocks))

    def position_to_block(self, position: int) -> Optional[int]:
        """Return the block index containing the normalized position (or None)."""
        idx = bisect_right(self._starts, position) - 1
        if idx < 0 or idx >= len(self.blocks):
            return None
        block = self.blocks[idx]
        return idx if position < block.end else None

    def dump_text(self, dest: Path) -> None:
        """Write the linearized text view to disk for inspection."""
        with dest.open("w", encoding="utf-8") as handle:
            current_page = -1
            for block in self.blocks:
                if block.page_index != current_page:
                    current_page = block.page_index
                    handle.write(f"\n=== Page {current_page + 1} ===\n")
                handle.write(block.text.rstrip() + "\n")


def load_questions(
    json_path: Path,
    *,
    subject: Optional[str],
    year: Optional[int],
    target: Optional[str],
    only_numbers: Optional[Sequence[int]],
) -> List[Question]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    questions: List[Question] = []
    filters_applied = []

    if subject:
        filters_applied.append(f"subject={subject}")
    if year is not None:
        filters_applied.append(f"year={year}")
    if target:
        filters_applied.append(f"target={target}")
    if only_numbers:
        filters_applied.append(f"numbers={sorted(only_numbers)}")

    logging.info(
        "Loading questions from %s%s",
        json_path,
        f" ({', '.join(filters_applied)})" if filters_applied else "",
    )

    number_filter = set(only_numbers) if only_numbers else None

    for entry in payload:
        if subject and entry.get("subject") != subject:
            continue
        if year is not None and entry.get("year") != year:
            continue
        if target and entry.get("target") != target:
            continue

        content = entry.get("content") or {}
        number = content.get("question_number")
        text = content.get("question_text")
        if number is None or text is None:
            continue
        try:
            number_int = int(number)
        except (TypeError, ValueError):
            logging.warning("Skipping question with non-integer number: %r", number)
            continue
        if number_filter and number_int not in number_filter:
            continue

        normalized = normalize_text(text)
        if not normalized:
            logging.warning("Skipping question %s (empty after normalization)", number)
            continue

        questions.append(
            Question(
                number=number_int,
                text=text,
                normalized=normalized,
                raw_entry=entry,
            )
        )

    questions.sort(key=lambda q: q.number)
    logging.info("Loaded %s questions", len(questions))
    return questions


@dataclass
class MatchResult:
    question: Question
    start_pos: int
    start_block_idx: int
    end_block_idx: int
    matched_length: int
    partial: bool


@dataclass
class MismatchDetail:
    question_number: int
    reason: str
    normalized_length: int
    closest_page: Optional[int]
    similarity: Optional[float]
    snippet: Optional[str]
    missing_words: List[str]


def find_match_position(
    normalized: str,
    text_stream: str,
    *,
    start_offset: int,
    min_prefix_ratio: float,
) -> tuple[int, int, bool]:
    """Return (hit_index, matched_length, partial_flag) with optional prefix matching."""
    if not normalized:
        return -1, 0, False

    # First attempt an exact match from the rolling cursor, then from the start.
    hit = text_stream.find(normalized, start_offset)
    if hit != -1:
        return hit, len(normalized), False

    hit = text_stream.find(normalized)
    if hit != -1:
        return hit, len(normalized), False

    # Fallback: try a prefix (e.g., 70% of the question) from the start of the string.
    prefix_len = max(1, math.ceil(len(normalized) * min_prefix_ratio))
    prefix = normalized[:prefix_len]

    hit = text_stream.find(prefix, start_offset)
    if hit == -1:
        hit = text_stream.find(prefix)

    if hit == -1:
        return -1, 0, False

    return hit, prefix_len, True


def diagnose_no_match(
    question: Question,
    index: LinearPdfIndex,
    *,
    reason: str,
) -> MismatchDetail:
    """Log additional context when a question cannot be matched to the PDF text."""
    if not index.blocks:
        logging.warning(
            "  Diagnostics unavailable: PDF yielded no text blocks (check extraction)."
        )
        return MismatchDetail(
            question_number=question.number,
            reason=reason,
            normalized_length=len(question.normalized),
            closest_page=None,
            similarity=None,
            snippet=None,
            missing_words=[],
        )

    best_ratio = 0.0
    best_block: Optional[Block] = None
    for block in index.blocks:
        ratio = SequenceMatcher(
            None,
            question.text.lower(),
            block.text.lower(),
        ).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_block = block

    if not best_block:
        logging.warning("  Diagnostics unavailable: no comparable text blocks found.")
        return MismatchDetail(
            question_number=question.number,
            reason=reason,
            normalized_length=len(question.normalized),
            closest_page=None,
            similarity=None,
            snippet=None,
            missing_words=[],
        )

    snippet = " ".join(best_block.text.split())
    if len(snippet) > 160:
        snippet = snippet[:157] + "..."

    logging.warning(
        "  Closest match is on page %s with similarity %.2f: %s",
        best_block.page_index + 1,
        best_ratio,
        snippet,
    )

    missing_words = sorted(
        set(question.text.lower().split()) - set(best_block.text.lower().split())
    )
    if missing_words:
        logging.warning(
            "  Words in JSON but absent from closest block: %s%s",
            ", ".join(missing_words[:10]),
            " (truncated)" if len(missing_words) > 10 else "",
        )
    return MismatchDetail(
        question_number=question.number,
        reason=reason,
        normalized_length=len(question.normalized),
        closest_page=best_block.page_index + 1 if best_block else None,
        similarity=best_ratio if best_block else None,
        snippet=snippet,
        missing_words=missing_words,
    )


def match_questions_to_blocks(
    index: LinearPdfIndex,
    questions: Sequence[Question],
    *,
    min_forward_offset: int = 0,
) -> Tuple[List[MatchResult], List[MismatchDetail]]:
    matches: List[MatchResult] = []
    search_cursor = min_forward_offset
    mismatches: List[MismatchDetail] = []
    previous_context_added: set[int] = set()

    def add_previous_context(current_idx: int) -> None:
        if current_idx <= 0:
            return
        prev_question = questions[current_idx - 1]
        if prev_question.number in previous_context_added:
            return
        mismatches.append(
            MismatchDetail(
                question_number=prev_question.number,
                reason="previous_of_mismatch",
                normalized_length=len(prev_question.normalized),
                closest_page=None,
                similarity=None,
                snippet=None,
                missing_words=[],
            )
        )
        previous_context_added.add(prev_question.number)

    for idx, question in enumerate(questions):
        normalized = question.normalized
        if not normalized:
            logging.warning("Question %s has no normalized text", question.number)
            mismatches.append(
                MismatchDetail(
                    question_number=question.number,
                    reason="empty_normalized_text",
                    normalized_length=0,
                    closest_page=None,
                    similarity=None,
                    snippet=None,
                    missing_words=[],
                )
            )
            add_previous_context(idx)
            continue

        hit, matched_len, partial = find_match_position(
            normalized,
            index.global_normalized,
            start_offset=search_cursor,
            min_prefix_ratio=0.2,
        )
        if hit == -1:
            logging.warning(
                "Question %s (%s chars) not found in PDF text stream",
                question.number,
                len(normalized),
            )
            mismatch_detail = diagnose_no_match(
                question,
                index,
                reason="not_found_in_stream",
            )
            mismatches.append(mismatch_detail)
            add_previous_context(idx)
            continue

        block_idx = index.position_to_block(hit)
        if block_idx is None:
            logging.warning(
                "Found position %s for question %s but no enclosing block",
                hit,
                question.number,
            )
            mismatch_detail = diagnose_no_match(
                question,
                index,
                reason="no_enclosing_block",
            )
            mismatches.append(mismatch_detail)
            add_previous_context(idx)
            continue

        if partial:
            coverage = matched_len / len(normalized) if len(normalized) else 0
            logging.warning(
                "Question %s matched only %.0f%% of its text; using prefix fallback.",
                question.number,
                coverage * 100,
            )

        matches.append(
            MatchResult(
                question=question,
                start_pos=hit,
                start_block_idx=block_idx,
                end_block_idx=block_idx,  # placeholder; updated below
                matched_length=matched_len,
                partial=partial,
            )
        )
        search_cursor = hit + matched_len

    # Determine end blocks by looking at the next match.
    total_blocks = len(index.blocks)
    for pos, match in enumerate(matches):
        if pos + 1 < len(matches):
            next_start_block = matches[pos + 1].start_block_idx
            match.end_block_idx = max(match.start_block_idx, next_start_block - 1)
        else:
            match.end_block_idx = total_blocks - 1

    logging.info("Matched %s/%s questions", len(matches), len(questions))
    return matches, mismatches


def extract_explanation_text(
    match: MatchResult,
    index: LinearPdfIndex,
    *,
    skip_texts: Optional[Iterable[str]] = None,
) -> str:
    """Extract explanation text following the options for a matched question."""
    blocks = index.blocks[match.start_block_idx : match.end_block_idx + 1]
    if not blocks:
        return ""

    content = match.question.raw_entry.get("content") or {}
    option_symbols: List[str] = []
    option_texts: List[str] = []
    for option in content.get("options") or []:
        symbol = (option.get("index") or "").strip()
        if symbol and symbol not in option_symbols:
            option_symbols.append(symbol)
        text = option.get("text")
        if isinstance(text, str):
            option_texts.append(text)

    def squeeze(text: str) -> str:
        return "".join(ch for ch in text if not ch.isspace())

    def alnum_only(text: str) -> str:
        return "".join(ch for ch in text if ch.isalnum())

    option_squeezed = [squeeze(text) for text in option_texts if text]
    option_alnum = [alnum_only(text) for text in option_texts if text]
    per_page_blocks: Dict[int, List[Tuple[int, Block]]] = defaultdict(list)
    for local_idx, block in enumerate(blocks):
        per_page_blocks[block.page_index].append((local_idx, block))

    table_block_indices: set[int] = set()
    for page_index, page_blocks in per_page_blocks.items():
        candidate_starts: List[float] = []
        flagged: set[int] = set()
        for i, (idx_a, block_a) in enumerate(page_blocks):
            rect_a = block_a.bbox
            for idx_b, block_b in page_blocks[i + 1 :]:
                rect_b = block_b.bbox
                overlap = min(rect_a.y1, rect_b.y1) - max(rect_a.y0, rect_b.y0)
                if overlap <= 0:
                    continue
                height_a = rect_a.y1 - rect_a.y0
                height_b = rect_b.y1 - rect_b.y0
                min_height = min(height_a, height_b)
                if min_height <= 0:
                    continue
                if overlap / min_height < 0.35:
                    continue
                if rect_a.x1 <= rect_b.x0:
                    gap = rect_b.x0 - rect_a.x1
                elif rect_b.x1 <= rect_a.x0:
                    gap = rect_a.x0 - rect_b.x1
                else:
                    continue
                if gap < 4.0:
                    continue
                flagged.update({idx_a, idx_b})
                candidate_starts.append(min(rect_a.y0, rect_b.y0))
        if candidate_starts and len(flagged) >= 2:
            threshold = max(min(candidate_starts) - 35.0, 0.0)
            for idx, block in page_blocks:
                if idx in flagged or block.bbox.y0 >= threshold:
                    table_block_indices.add(idx)

    normalized_page_counts = getattr(index, "_normalized_page_counts", None)
    if normalized_page_counts is None:
        page_sets: Dict[str, set[int]] = defaultdict(set)
        for block in index.blocks:
            if block.normalized:
                page_sets[block.normalized].add(block.page_index)
        normalized_page_counts = {
            key: len(pages) for key, pages in page_sets.items() if pages
        }
        setattr(index, "_normalized_page_counts", normalized_page_counts)

    lines: List[Tuple[str, int]] = []
    for local_idx, block in enumerate(blocks):
        for raw_line in block.text.splitlines():
            lines.append((raw_line, local_idx))

    skip_texts = [text for text in (skip_texts or []) if text]
    skip_set = {text.strip() for text in skip_texts if text.strip()}
    skip_values = tuple(skip_set)
    skip_squeezed = [squeeze(text) for text in skip_texts if text]
    skip_alnum = [alnum_only(text) for text in skip_texts if text]

    line_infos: List[Tuple[str, str, str, int, int, int]] = []
    squeezed_parts: List[str] = []
    cursor = 0
    for raw_line, block_idx in lines:
        squeezed_line = squeeze(raw_line)
        alnum_line = alnum_only(raw_line)
        start = cursor
        cursor += len(squeezed_line)
        line_infos.append(
            (raw_line, squeezed_line, alnum_line, start, cursor, block_idx)
        )
        squeezed_parts.append(squeezed_line)

    squeezed_total = "".join(squeezed_parts)

    def slice_after_squeezed(raw_line: str, squeezed_count: int) -> str:
        if squeezed_count <= 0:
            return raw_line
        consumed = 0
        for idx, ch in enumerate(raw_line):
            if ch.isspace():
                continue
            consumed += 1
            if consumed == squeezed_count:
                return raw_line[idx + 1 :]
        return ""

    last_option_end = -1
    search_cursor = 0
    for candidate in option_squeezed:
        if not candidate:
            continue
        hit = squeezed_total.find(candidate, search_cursor)
        if hit == -1:
            continue
        end = hit + len(candidate)
        last_option_end = max(last_option_end, end)
        search_cursor = end

    start_line_idx = 0
    remainder_line: Optional[Tuple[str, int]] = None
    if last_option_end >= 0 and line_infos:
        for idx, info in enumerate(line_infos):
            raw_line, squeezed_line, _alnum_line, start, end, block_idx = info
            if last_option_end <= start:
                start_line_idx = idx
                break
            if last_option_end <= end:
                offset = last_option_end - start
                if offset < len(squeezed_line):
                    remainder_line = (
                        slice_after_squeezed(raw_line, offset),
                        block_idx,
                    )
                start_line_idx = idx + 1
                break
        else:
            start_line_idx = len(line_infos)

    explanation_lines: List[str] = []
    candidate_pairs: List[Tuple[str, str]] = []
    seen_candidates: set[Tuple[str, str]] = set()
    for pair in zip(option_squeezed, option_alnum):
        if pair not in seen_candidates:
            candidate_pairs.append(pair)
            seen_candidates.add(pair)
    for pair in zip(skip_squeezed, skip_alnum):
        if pair not in seen_candidates:
            candidate_pairs.append(pair)
            seen_candidates.add(pair)

    pending_remainders: List[Tuple[str, str]] = []

    def process_line(raw_line: str, block_idx: Optional[int]) -> None:
        nonlocal pending_remainders
        stripped = raw_line.strip()
        squeezed = squeeze(stripped)
        alnum = alnum_only(stripped)

        if block_idx is not None and block_idx in table_block_indices:
            return

        if block_idx is not None and 0 <= block_idx < len(blocks):
            block = blocks[block_idx]
            if normalized_page_counts.get(block.normalized, 0) >= 3:
                return

        if stripped and any(
            stripped == symbol or stripped == f"{symbol}."
            for symbol in option_symbols
        ):
            return

        if stripped.startswith("◤"):
            return

        if stripped.startswith("-") and stripped.endswith("-"):
            inner = stripped[1:-1].strip()
            if inner.replace(" ", "").isdigit():
                return

        pending_matched = False
        updated_pending: List[Tuple[str, str]] = []
        for remaining_sq, remaining_al in pending_remainders:
            if squeezed and remaining_sq.startswith(squeezed):
                new_sq = remaining_sq[len(squeezed) :]
                new_al = (
                    remaining_al[len(alnum) :]
                    if alnum and remaining_al.startswith(alnum)
                    else remaining_al
                )
                if new_sq.strip() or new_al.strip():
                    updated_pending.append((new_sq, new_al))
                pending_matched = True
            elif alnum and remaining_al.startswith(alnum):
                new_al = remaining_al[len(alnum) :]
                new_sq = remaining_sq
                if new_sq.strip() or new_al.strip():
                    updated_pending.append((new_sq, new_al))
                pending_matched = True
            else:
                updated_pending.append((remaining_sq, remaining_al))
        if pending_matched:
            pending_remainders = updated_pending
            return

        if skip_set and stripped in skip_set:
            return

        def overlaps_substantially(text: str, candidates: Sequence[str]) -> bool:
            text_len = len(text)
            if text_len == 0:
                return False
            for candidate in candidates:
                if not candidate:
                    continue
                cand_len = len(candidate)
                if text == candidate:
                    return True
                if text_len >= 6 and candidate.startswith(text):
                    return True
                if cand_len >= 6 and text.startswith(candidate):
                    coverage = cand_len / max(text_len, 1)
                    if coverage >= 0.6:
                        return True
                if text_len < 6 or cand_len < 6:
                    continue
                shorter = min(text_len, cand_len)
                longer = max(text_len, cand_len)
                if shorter / longer >= 0.6 and (
                    text in candidate or candidate in text
                ):
                    return True
            return False

        skip_line = False
        if stripped and overlaps_substantially(stripped, skip_values):
            skip_line = True
        if (
            not skip_line
            and squeezed
            and overlaps_substantially(squeezed, option_squeezed)
        ):
            skip_line = True
        if (
            not skip_line
            and squeezed
            and overlaps_substantially(squeezed, skip_squeezed)
        ):
            skip_line = True
        if (
            not skip_line
            and alnum
            and overlaps_substantially(alnum, option_alnum)
        ):
            skip_line = True
        if (
            not skip_line
            and alnum
            and overlaps_substantially(alnum, skip_alnum)
        ):
            skip_line = True
        if not skip_line and alnum and len(alnum) >= 6:
            for candidate in option_alnum:
                if not candidate:
                    continue
                shorter = min(len(alnum), len(candidate))
                if shorter < 6:
                    continue
                longer = max(len(alnum), len(candidate))
                if longer > shorter * 1.5:
                    continue
                if SequenceMatcher(None, alnum, candidate).ratio() >= 0.9:
                    skip_line = True
                    break
            if not skip_line:
                for candidate in skip_alnum:
                    if not candidate:
                        continue
                    shorter = min(len(alnum), len(candidate))
                    if shorter < 6:
                        continue
                    longer = max(len(alnum), len(candidate))
                    if longer > shorter * 1.5:
                        continue
                    if SequenceMatcher(None, alnum, candidate).ratio() >= 0.9:
                        skip_line = True
                        break
        if skip_line:
            for cand_sq, cand_al in candidate_pairs:
                if not cand_sq and not cand_al:
                    continue
                remainder_sq = ""
                remainder_al = ""
                matched = False
                if squeezed and cand_sq and cand_sq.startswith(squeezed):
                    remainder_sq = cand_sq[len(squeezed) :]
                    if cand_al and alnum and cand_al.startswith(alnum):
                        remainder_al = cand_al[len(alnum) :]
                    else:
                        remainder_al = cand_al or ""
                    matched = True
                elif alnum and cand_al and cand_al.startswith(alnum):
                    remainder_al = cand_al[len(alnum) :]
                    if cand_sq and squeezed and cand_sq.startswith(squeezed):
                        remainder_sq = cand_sq[len(squeezed) :]
                    else:
                        remainder_sq = cand_sq or ""
                    matched = True
                elif squeezed and cand_sq and squeezed.endswith(cand_sq):
                    matched = True
                elif alnum and cand_al and alnum.endswith(cand_al):
                    matched = True
                if matched:
                    if remainder_sq.strip() or remainder_al.strip():
                        pending_remainders.append((remainder_sq, remainder_al))
                    break
            return

        explanation_lines.append(raw_line.rstrip())

    if remainder_line and remainder_line[0].strip():
        process_line(remainder_line[0], remainder_line[1])

    for raw_line, _squeezed, _alnum, _start, _end, block_idx in line_infos[
        start_line_idx:
    ]:
        process_line(raw_line, block_idx)

    while explanation_lines and not explanation_lines[0].strip():
        explanation_lines.pop(0)
    while explanation_lines and not explanation_lines[-1].strip():
        explanation_lines.pop()

    return "\n".join(explanation_lines)


def bbox_pp_to_camelot(bbox: Tuple[float, float, float, float], page_height: float) -> str:
    x0, top, x1, bottom = bbox
    y_top = page_height - top
    y_bottom = page_height - bottom
    return f"{x0},{y_top},{x1},{y_bottom}"


def bbox_intersects(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or ax0 >= bx1 or ay1 <= by0 or ay0 >= by1)


def compute_union_bbox(rects: Iterable[fitz.Rect]) -> Optional[Tuple[float, float, float, float]]:
    rect_list = list(rects)
    if not rect_list:
        return None
    x0 = min(r.x0 for r in rect_list)
    y0 = min(r.y0 for r in rect_list)
    x1 = max(r.x1 for r in rect_list)
    y1 = max(r.y1 for r in rect_list)
    return (x0, y0, x1, y1)


def pad_bbox(
    bbox: Tuple[float, float, float, float],
    pad: float,
    *,
    max_width: Optional[float] = None,
    max_height: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    x0 = max(0.0, x0 - pad)
    y0 = max(0.0, y0 - pad)
    x1 = x1 + pad
    y1 = y1 + pad
    if max_width is not None:
        x1 = min(max_width, x1)
    if max_height is not None:
        y1 = min(max_height, y1)
    return (x0, y0, x1, y1)


def camelot_table_to_rows(table) -> List[List[str]]:
    if pd is None:
        return [[(" ".join(str(cell).split()) if cell else "") for cell in row] for row in table.data]
    df = table.df.copy()
    df = df.applymap(lambda value: " ".join(str(value).split()) if pd.notna(value) else "")
    return df.values.tolist()


def read_camelot_tables(pdf_path: Path, page_index: int, region: Optional[str]):
    if camelot is None:
        return []
    pages_str = str(page_index + 1)
    lattice_kwargs = dict(
        filepath=str(pdf_path),
        pages=pages_str,
        flavor="lattice",
        strip_text=" \n",
        line_scale=40,
    )
    if region:
        lattice_kwargs["table_regions"] = [region]
    try:
        tables = camelot.read_pdf(**lattice_kwargs)
    except Exception as exc:  # pylint: disable=broad-except
        logging.debug("Camelot lattice failed on page %s: %s", page_index + 1, exc)
        tables = []
    if len(tables) == 0:
        stream_kwargs = dict(
            filepath=str(pdf_path),
            pages=pages_str,
            flavor="stream",
            strip_text=" \n",
            row_tol=10,
            column_tol=10,
        )
        if region:
            stream_kwargs["table_regions"] = [region]
        try:
            tables = camelot.read_pdf(**stream_kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            logging.debug("Camelot stream failed on page %s: %s", page_index + 1, exc)
            tables = []
    return tables


def cleanup_camelot_tables(tables: Optional[Sequence]) -> None:
    """Best-effort cleanup for Camelot table objects to release temp files on Windows."""
    if not tables:
        return
    for table in tables:
        parser = getattr(table, "_parser", None)
        if parser is not None:
            for attr in ("fp", "_fp", "file", "f", "stream"):
                stream = getattr(parser, attr, None)
                if stream and hasattr(stream, "close"):
                    try:
                        stream.close()
                    except Exception:  # pylint: disable=broad-except
                        pass
            close_method = getattr(parser, "close", None)
            if callable(close_method):
                try:
                    close_method()
                except Exception:  # pylint: disable=broad-except
                    pass
        temp_pdf = None
        try:
            temp_pdf = table.parsing_report.get("temp_pdf")  # type: ignore[attr-defined]
        except Exception:  # pylint: disable=broad-except
            temp_pdf = None
        if temp_pdf:
            temp_path = Path(temp_pdf)
            temp_dir = temp_path.parent
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:  # pylint: disable=broad-except
                pass
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
                if hasattr(atexit, "_exithandlers"):
                    handlers = list(getattr(atexit, "_exithandlers", []))
                    updated = [
                        (fn, args, kwargs)
                        for (fn, args, kwargs) in handlers
                        if not (
                            fn is shutil.rmtree
                            and args
                            and Path(args[0]) == temp_dir
                        )
                    ]
                    if len(updated) != len(handlers):
                        setattr(atexit, "_exithandlers", updated)
            except Exception:  # pylint: disable=broad-except
                pass
    try:
        gc.collect()
    except Exception:  # pylint: disable=broad-except
        pass


@contextmanager
def cropped_pdf(
    pdf_path: Path,
    page_index: int,
    bbox_pp: Tuple[float, float, float, float],
):
    """Yield path to a temporary single-page PDF cropped to bbox_pp (points)."""
    if PdfReader is None or PdfWriter is None:
        logging.debug("pypdf unavailable; skipping cropped PDF helper.")
        yield None
        return
    writer: Optional[PdfWriter] = None  # type: ignore[assignment]
    source_handle = pdf_path.open("rb")
    try:
        reader = PdfReader(source_handle)
        page = reader.pages[page_index]
        page_height = float(page.mediabox.height)

        x0, top, x1, bottom = bbox_pp
        lower_left = (x0, page_height - bottom)
        upper_right = (x1, page_height - top)

        page.cropbox.lower_left = lower_left
        page.cropbox.upper_right = upper_right
        page.mediabox.lower_left = lower_left
        page.mediabox.upper_right = upper_right

        writer = PdfWriter()
        writer.add_page(page)
    finally:
        source_handle.close()

    if writer is None:
        logging.debug("Failed to initialize PdfWriter; skipping cropped PDF helper.")
        yield None
        return

    with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        writer.write(tmp)
        tmp_path = Path(tmp.name)

    try:
        yield tmp_path
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def _lines_from_chars(chars, y_tol=2.0):
    if not chars:
        return []
    chars_sorted = sorted(chars, key=lambda c: (round(c["top"], 1), c["x0"]))
    lines, current = [], [chars_sorted[0]]
    for ch in chars_sorted[1:]:
        if abs(ch["top"] - current[-1]["top"]) <= y_tol:
            current.append(ch)
        else:
            lines.append(current)
            current = [ch]
    lines.append(current)
    for ln in lines:
        ln.sort(key=lambda c: c["x0"])
    return lines


def _join_line_preserve_combining(line_chars, gap_space=1.5):
    s: List[str] = []
    prev_right = None
    for ch in line_chars:
        txt = ch.get("text", "")
        if not txt:
            continue
        if unicodedata.category(txt[0]) == "Mn" and s:
            s[-1] = s[-1] + txt
        else:
            if prev_right is not None and (ch["x0"] - prev_right) > gap_space:
                s.append(" ")
            s.append(txt)
        prev_right = ch["x1"]
    return unicodedata.normalize("NFC", "".join(s)).strip()


def _estimate_column_split(chars, fallback_split=200.0):
    xs = sorted({round(ch["x0"], 2) for ch in chars if ch.get("text", "").strip()})
    if len(xs) < 2:
        return fallback_split
    gaps = [(xs[i + 1] - xs[i], i) for i in range(len(xs) - 1)]
    gap, idx = max(gaps, key=lambda item: item[0])
    if gap < 10:
        return fallback_split
    return (xs[idx] + xs[idx + 1]) / 2


def extract_table_with_chars(
    page,
    bbox_pp: Tuple[float, float, float, float],
    *,
    y_tol: float = 2.0,
    gap_space: float = 1.5,
    join_lines_with: str = " ",
) -> List[List[str]]:
    cropped = page.crop(bbox_pp)
    chars = [ch for ch in cropped.chars if ch.get("text", "").strip()]
    if not chars:
        return []

    split_x = _estimate_column_split(chars)
    lines = _lines_from_chars(chars, y_tol=y_tol)

    rows: List[List[str]] = []
    current_left: List[str] = []
    current_right: List[str] = []
    right_joiner = join_lines_with if join_lines_with in (" ", "\n") else " "

    def append_row():
        if current_left or current_right:
            left_text = (" ".join(current_left)).strip()
            right_text = right_joiner.join(current_right).strip()
            rows.append([left_text, right_text])
            current_left.clear()
            current_right.clear()

    for line_chars in lines:
        left_chars = [ch for ch in line_chars if ch["x0"] < split_x]
        right_chars = [ch for ch in line_chars if ch["x0"] >= split_x]
        left_text = _join_line_preserve_combining(left_chars, gap_space=gap_space) if left_chars else ""
        right_text = _join_line_preserve_combining(right_chars, gap_space=gap_space) if right_chars else ""
        left_text = left_text.strip()
        right_text = right_text.strip()
        if left_text:
            append_row()
            current_left.append(left_text)
        if right_text:
            current_right.append(right_text)

    append_row()
    return rows


def extract_rows_from_region(
    pdf_path: Path,
    page_index: int,
    base_bbox: Tuple[float, float, float, float],
    *,
    base_page=None,
    expand_left: bool = True,
    expand_right: bool = True,
    crop_page: bool = True,
    y_tol: float = 2.0,
    gap_space: float = 1.5,
    join_lines_with: str = " ",
) -> Tuple[List[List[str]], Tuple[float, float, float, float]]:
    if not PDFPLUMBER_AVAILABLE or pdfplumber is None:
        return [], base_bbox

    close_doc = False
    page = base_page
    if page is None:
        plumber_doc = pdfplumber.open(str(pdf_path))
        page = plumber_doc.pages[page_index]
        close_doc = True

    try:
        x0, top, x1, bottom = base_bbox
        if expand_left:
            x0 = 0.0
        if expand_right:
            x1 = page.width
        expanded_bbox = (x0, top, x1, bottom)

        rows: List[List[str]] = []

        if CAMEL0T_AVAILABLE and camelot is not None:
            camelot_tables: List = []
            camelot_rows: Optional[List[List[str]]] = None
            if crop_page:
                with cropped_pdf(pdf_path, page_index, expanded_bbox) as cropped_path:
                    if cropped_path is not None:
                        camelot_tables = read_camelot_tables(cropped_path, 0, None)
            if not crop_page or not camelot_tables:
                region = bbox_pp_to_camelot(expanded_bbox, page.height)
                camelot_tables = read_camelot_tables(pdf_path, page_index, region)

            if len(camelot_tables) == 1:
                camelot_rows = camelot_table_to_rows(camelot_tables[0])
                if not any(any(cell for cell in row) for row in camelot_rows):
                    camelot_rows = None
            cleanup_camelot_tables(camelot_tables)
            if camelot_rows is not None:
                return camelot_rows, expanded_bbox

        rows = extract_table_with_chars(
            page,
            expanded_bbox,
            y_tol=y_tol,
            gap_space=gap_space,
            join_lines_with=join_lines_with,
        )
        return rows, expanded_bbox
    finally:
        if close_doc:
            plumber_doc.close()


def load_tables_for_page(
    pdf_path: Path,
    page_index: int,
    *,
    table_opts: Dict[str, object],
) -> List[Dict[str, object]]:
    if not PDFPLUMBER_AVAILABLE or pdfplumber is None:
        return []

    extracted: List[Dict[str, object]] = []
    try:
        with pdfplumber.open(str(pdf_path)) as plumber_doc:
            page = plumber_doc.pages[page_index]
            candidate_tables = page.find_tables() or []
            for tbl in candidate_tables:
                rows, region_bbox = extract_rows_from_region(
                    pdf_path,
                    page_index,
                    tbl.bbox,
                    base_page=page,
                    expand_left=bool(table_opts.get("expand_left", True)),
                    expand_right=bool(table_opts.get("expand_right", True)),
                    crop_page=bool(table_opts.get("crop_page", True)),
                    y_tol=table_opts.get("y_tol", 2.0),
                    gap_space=table_opts.get("gap_space", 1.5),
                    join_lines_with=table_opts.get("join_lines_with", " "),
                )
                if not rows:
                    continue
                extracted.append(
                    {
                        "page": page_index + 1,
                        "bbox": [round(coord, 2) for coord in region_bbox],
                        "rows": rows,
                    }
                )
    except Exception as exc:  # pylint: disable=broad-except
        logging.debug("Failed to extract tables on page %s: %s", page_index + 1, exc)
    return extracted


def extract_tables_for_match(
    pdf_path: Path,
    match: MatchResult,
    index: LinearPdfIndex,
    cache: Dict[Tuple[int, bool, bool, bool], List[Dict[str, object]]],
    table_opts: Dict[str, object],
) -> List[Dict[str, object]]:
    if not PDFPLUMBER_AVAILABLE:
        logging.debug("Table extraction unavailable; dependencies missing.")
        return []

    blocks = index.blocks[match.start_block_idx : match.end_block_idx + 1]
    if not blocks:
        return []

    per_page: Dict[int, List[fitz.Rect]] = defaultdict(list)
    for block in blocks:
        per_page[block.page_index].append(block.bbox)

    referenced: List[Dict[str, object]] = []
    for page_index, rects in per_page.items():
        cache_key = (
            page_index,
            bool(table_opts.get("expand_left", True)),
            bool(table_opts.get("expand_right", True)),
            bool(table_opts.get("crop_page", True)),
        )
        tables = cache.get(cache_key)
        if tables is None:
            tables = load_tables_for_page(
                pdf_path,
                page_index,
                table_opts=table_opts,
            )
            cache[cache_key] = tables
        if not tables:
            continue
        union_bbox = compute_union_bbox(rects)
        if union_bbox is None:
            continue
        for table_info in tables:
            table_bbox = tuple(table_info["bbox"])  # type: ignore[arg-type]
            padded = pad_bbox(union_bbox, pad=20.0)
            if bbox_intersects(padded, table_bbox):  # type: ignore[arg-type]
                referenced.append(table_info)

    return referenced


def union_rectangles(rects: Iterable[fitz.Rect], padding: float, page_rect: fitz.Rect) -> fitz.Rect:
    rect_list = list(rects)
    if not rect_list:
        return fitz.Rect()

    x0 = min(r.x0 for r in rect_list)
    y0 = min(r.y0 for r in rect_list)
    x1 = max(r.x1 for r in rect_list)
    y1 = max(r.y1 for r in rect_list)

    expanded = fitz.Rect(
        x0 - padding,
        y0 - padding,
        x1 + padding,
        y1 + padding,
    )
    return fitz.Rect(
        max(expanded.x0, page_rect.x0),
        max(expanded.y0, page_rect.y0),
        min(expanded.x1, page_rect.x1),
        min(expanded.y1, page_rect.y1),
    )


def annotate_pdf(
    doc: fitz.Document,
    index: LinearPdfIndex,
    matches: Sequence[MatchResult],
    *,
    padding: float,
    stroke_width: float,
    label_font_size: float,
    label_prefix: str,
    text_offset: float,
) -> None:
    for match in matches:
        slice_blocks = index.blocks[match.start_block_idx : match.end_block_idx + 1]
        if not slice_blocks:
            continue
        per_page: defaultdict[int, List[fitz.Rect]] = defaultdict(list)
        for block in slice_blocks:
            per_page[block.page_index].append(block.bbox)

        for page_index, rects in per_page.items():
            page = doc[page_index]
            box = union_rectangles(rects, padding, page.rect)
            if box.is_empty:
                logging.debug(
                    "Empty rectangle for question %s on page %s",
                    match.question.number,
                    page_index + 1,
                )
                continue

            shape = page.new_shape()
            shape.draw_rect(box)
            shape.finish(color=(1, 0, 0), width=stroke_width)
            shape.commit()

            label_text = f"{label_prefix}{match.question.number}"
            anchor_y = max(box.y0 - text_offset, 10)
            anchor = fitz.Point(box.x0, anchor_y)
            page.insert_text(
                anchor,
                label_text,
                fontsize=label_font_size,
                color=(1, 0, 0),
                fontname="helv",
            )
            logging.debug(
                "Annotated question %s on page %s",
                match.question.number,
                page_index + 1,
            )


def launch_omit_gui(doc: fitz.Document, *, zoom: float = 1.0) -> List[OmitRegion]:
    """Interactive GUI for selecting regions to exclude from processing."""
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except ImportError:
        logging.error("Tkinter is not available; cannot launch omit-region GUI.")
        return []

    if doc.page_count == 0:
        logging.warning("PDF has no pages; skipping omit-region GUI.")
        return []

    class OmitGUI:
        def __init__(self, root: tk.Tk) -> None:
            self.root = root
            self.root.title("Select Header/Footer Regions")
            self.doc = doc
            self.zoom = zoom
            self.page_index = 0
            self.image_cache: Optional[tk.PhotoImage] = None
            self.canvas = tk.Canvas(self.root, highlightthickness=0)
            self.canvas.grid(row=1, column=0, columnspan=4, sticky="nsew")
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(1, weight=1)

            self.instructions = ttk.Label(
                self.root,
                text="Drag to draw rectangles to omit. Scroll to zoom. Use Next/Prev to change pages.",
            )
            self.instructions.grid(row=0, column=0, columnspan=3, padx=6, pady=6, sticky="w")

            self.apply_all_var = tk.BooleanVar(value=True)
            self.apply_all_check = ttk.Checkbutton(
                self.root,
                text="Apply to all pages",
                variable=self.apply_all_var,
            )
            self.apply_all_check.grid(row=0, column=3, padx=6, pady=6, sticky="e")

            self.page_label = ttk.Label(self.root, text="")
            self.page_label.grid(row=2, column=0, padx=6, pady=6, sticky="w")

            self.prev_btn = ttk.Button(self.root, text="◀ Prev", command=self.prev_page)
            self.prev_btn.grid(row=2, column=1, padx=6, pady=6, sticky="e")

            self.next_btn = ttk.Button(self.root, text="Next ▶", command=self.next_page)
            self.next_btn.grid(row=2, column=2, padx=6, pady=6, sticky="w")

            self.clear_btn = ttk.Button(
                self.root, text="Remove Last", command=self.remove_last
            )
            self.clear_btn.grid(row=2, column=3, padx=6, pady=6, sticky="e")

            self.finish_btn = ttk.Button(self.root, text="Finish", command=self.finish)
            self.finish_btn.grid(row=3, column=2, padx=6, pady=6, sticky="e")

            self.cancel_btn = ttk.Button(self.root, text="Cancel", command=self.cancel)
            self.cancel_btn.grid(row=3, column=1, padx=6, pady=6, sticky="w")

            self.status_var = tk.StringVar(value="")
            self.status_label = ttk.Label(self.root, textvariable=self.status_var)
            self.status_label.grid(row=3, column=0, padx=6, pady=6, sticky="w")

            self.canvas.bind("<ButtonPress-1>", self.on_press)
            self.canvas.bind("<B1-Motion>", self.on_drag)
            self.canvas.bind("<ButtonRelease-1>", self.on_release)
            self.canvas.bind("<MouseWheel>", self.on_scroll)  # Windows / macOS
            self.canvas.bind("<Button-4>", self.on_scroll)    # Linux scroll up
            self.canvas.bind("<Button-5>", self.on_scroll)    # Linux scroll down

            self.start_x: Optional[float] = None
            self.start_y: Optional[float] = None
            self.current_rect_id: Optional[int] = None
            self.cancelled = False
            self.finished = False
            self.regions: Dict[int, List[fitz.Rect]] = defaultdict(list)
            self.canvas_region_ids: Dict[int, List[int]] = defaultdict(list)
            self.region_actions: List[List[Tuple[int, fitz.Rect]]] = []

            self.load_page()

        def load_page(self) -> None:
            page = self.doc[self.page_index]
            pix = page.get_pixmap(matrix=fitz.Matrix(self.zoom, self.zoom), alpha=False)
            data = pix.tobytes("ppm")
            self.image_cache = tk.PhotoImage(data=data)
            self.canvas.delete("all")
            self.canvas.config(width=pix.width, height=pix.height)
            self.canvas.create_image(0, 0, anchor="nw", image=self.image_cache)
            ids: List[int] = []
            for rect in self.regions.get(self.page_index, []):
                rect_id = self.canvas.create_rectangle(
                    rect.x0 * self.zoom,
                    rect.y0 * self.zoom,
                    rect.x1 * self.zoom,
                    rect.y1 * self.zoom,
                    outline="#ff6600",
                    width=2,
                )
                ids.append(rect_id)
            self.canvas_region_ids[self.page_index] = ids
            self.page_label.config(
                text=f"Page {self.page_index + 1} / {self.doc.page_count} | Zoom {self.zoom * 100:.0f}%"
            )
            self.update_status()

        def update_status(self) -> None:
            count = len(self.regions.get(self.page_index, []))
            total = sum(len(items) for items in self.regions.values())
            self.status_var.set(f"Page regions: {count} | Total regions: {total}")

        def on_press(self, event: tk.Event) -> None:
            self.start_x = event.x
            self.start_y = event.y
            self.current_rect_id = self.canvas.create_rectangle(
                event.x,
                event.y,
                event.x,
                event.y,
                outline="#ff6600",
                width=2,
            )

        def on_drag(self, event: tk.Event) -> None:
            if self.current_rect_id is not None and self.start_x is not None:
                self.canvas.coords(
                    self.current_rect_id,
                    self.start_x,
                    self.start_y,
                    event.x,
                    event.y,
                )

        def on_release(self, event: tk.Event) -> None:
            if (
                self.current_rect_id is None
                or self.start_x is None
                or self.start_y is None
            ):
                return
            x0, y0 = self.start_x, self.start_y
            x1, y1 = event.x, event.y
            if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
                self.canvas.delete(self.current_rect_id)
                self.current_rect_id = None
                self.start_x = None
                self.start_y = None
                return
            rect = fitz.Rect(
                min(x0, x1) / self.zoom,
                min(y0, y1) / self.zoom,
                max(x0, x1) / self.zoom,
                max(y0, y1) / self.zoom,
            )
            rect_id = self.current_rect_id
            pages = (
                list(range(self.doc.page_count))
                if self.apply_all_var.get()
                else [self.page_index]
            )
            stored_rects: List[Tuple[int, fitz.Rect]] = []
            for page_idx in pages:
                rect_copy = fitz.Rect(rect)
                self.regions[page_idx].append(rect_copy)
                stored_rects.append((page_idx, rect_copy))
                if page_idx == self.page_index and rect_id is not None:
                    self.canvas_region_ids[self.page_index].append(rect_id)
            self.region_actions.append(stored_rects)
            self.current_rect_id = None
            self.start_x = None
            self.start_y = None
            self.update_status()

        def remove_last(self) -> None:
            if not self.region_actions:
                return
            stored = self.region_actions.pop()
            for page_idx, rect_obj in stored:
                rect_list = self.regions.get(page_idx)
                if rect_list and rect_obj in rect_list:
                    rect_list.remove(rect_obj)
                    if not rect_list:
                        self.regions.pop(page_idx, None)
            self.load_page()

        def prev_page(self) -> None:
            if self.page_index == 0:
                return
            self.page_index -= 1
            self.load_page()

        def next_page(self) -> None:
            if self.page_index + 1 >= self.doc.page_count:
                return
            self.page_index += 1
            self.load_page()

        def finish(self) -> None:
            self.finished = True
            self.root.quit()

        def cancel(self) -> None:
            if messagebox.askyesno("Cancel", "Discard omit regions and exit?"):
                self.cancelled = True
                self.root.quit()

        def collect(self) -> List[OmitRegion]:
            result: List[OmitRegion] = []
            for page_idx, entries in self.regions.items():
                for rect in entries:
                    result.append(OmitRegion(page_idx, rect))
            return result

        def on_scroll(self, event: tk.Event) -> None:
            delta = 0
            if hasattr(event, "delta") and event.delta:
                delta = event.delta
            elif hasattr(event, "num"):
                if event.num == 4:
                    delta = 120
                elif event.num == 5:
                    delta = -120
            if delta == 0:
                return
            factor = 1.15 if delta > 0 else 1 / 1.15
            new_zoom = max(0.5, min(3.0, self.zoom * factor))
            if abs(new_zoom - self.zoom) < 1e-3:
                return
            self.zoom = new_zoom
            self.load_page()

    root = tk.Tk()
    gui = OmitGUI(root)
    root.mainloop()
    try:
        root.destroy()
    except Exception:
        pass
    if gui.cancelled:
        logging.info("Omit-region GUI cancelled; proceeding without exclusions.")
        return []
    regions = gui.collect()
    logging.info("Omit-region GUI captured %s regions.", len(regions))
    return regions


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate question regions in a PDF using metadata from a JSON file."
    )
    parser.add_argument("--pdf", required=True, type=Path, help="Input PDF to annotate.")
    parser.add_argument("--json", required=True, type=Path, help="Question metadata JSON.")
    parser.add_argument("--output", required=True, type=Path, help="Destination PDF path.")
    parser.add_argument("--subject", help="Filter questions by subject.")
    parser.add_argument("--year", type=int, help="Filter questions by exam year.")
    parser.add_argument("--target", help="Filter questions by target/audience.")
    parser.add_argument(
        "--question",
        type=int,
        nargs="+",
        help="Limit to specific question numbers (space-separated).",
    )
    parser.add_argument(
        "--dump-text",
        type=Path,
        help="Optional path to write the linearized PDF text stream.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=4.0,
        help="Extra padding (points) around detected regions.",
    )
    parser.add_argument(
        "--stroke-width",
        type=float,
        default=0.8,
        help="Rectangle stroke width (points).",
    )
    parser.add_argument(
        "--label-font-size",
        type=float,
        default=8.0,
        help="Label font size (points).",
    )
    parser.add_argument(
        "--label-prefix",
        default="Q",
        help="Prefix to place before each question number label.",
    )
    parser.add_argument(
        "--label-offset",
        type=float,
        default=6.0,
        help="Vertical offset (points) between the rectangle and the label baseline.",
    )
    parser.add_argument(
        "--expand-left",
        dest="expand_left",
        action="store_true",
        default=True,
        help="Force table region to the left page edge (default: on).",
    )
    parser.add_argument(
        "--no-expand-left",
        dest="expand_left",
        action="store_false",
        help="Do not force table region to the left page edge.",
    )
    parser.add_argument(
        "--expand-right",
        dest="expand_right",
        action="store_true",
        default=True,
        help="Force table region to the right page edge (default: on).",
    )
    parser.add_argument(
        "--no-expand-right",
        dest="expand_right",
        action="store_false",
        help="Do not force table region to the right page edge.",
    )
    parser.add_argument(
        "--crop-page",
        dest="crop_page",
        action="store_true",
        default=True,
        help="Physically crop table region before Camelot extraction (default: on).",
    )
    parser.add_argument(
        "--no-crop-page",
        dest="crop_page",
        action="store_false",
        help="Do not crop the page before Camelot extraction.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output PDF.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for troubleshooting.",
    )
    parser.add_argument(
        "--mismatch-report",
        type=Path,
        help="Optional path to write JSON diagnostics for unmatched questions.",
    )
    parser.add_argument(
        "--explanation-json",
        type=Path,
        help="Write filtered questions with extracted explanations to this JSON file.",
    )
    parser.add_argument(
        "--omit-gui",
        action="store_true",
        help="Launch a GUI to select header/footer regions to omit during processing.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.output.exists() and not args.overwrite:
        logging.error(
            "Output file %s already exists (use --overwrite to replace it).",
            args.output,
        )
        return 1

    if not args.pdf.exists():
        logging.error("PDF file not found: %s", args.pdf)
        return 1

    if not args.json.exists():
        logging.error("JSON file not found: %s", args.json)
        return 1

    questions = load_questions(
        args.json,
        subject=args.subject,
        year=args.year,
        target=args.target,
        only_numbers=args.question,
    )
    if not questions:
        logging.error("No questions matched the provided filters.")
        return 1

    with fitz.open(args.pdf) as doc:
        omit_regions: List[OmitRegion] = []
        if args.omit_gui:
            omit_regions = launch_omit_gui(doc)
        index = LinearPdfIndex(doc, omit_regions=omit_regions)
        if omit_regions:
            logging.info(
                "Omitting %s regions across %s pages",
                len(omit_regions),
                len({region.page_index for region in omit_regions}),
            )

        if args.dump_text:
            logging.info("Writing linearized text to %s", args.dump_text)
            index.dump_text(args.dump_text)

        matches, mismatches = match_questions_to_blocks(index, questions)

        table_cache: Dict[Tuple[int, bool, bool, bool], List[Dict[str, object]]] = {}
        table_opts = {
            "expand_left": args.expand_left,
            "expand_right": args.expand_right,
            "crop_page": args.crop_page,
            "y_tol": 2.0,
            "gap_space": 1.5,
            "join_lines_with": " ",
        }
        # Clear any existing explanation fields before repopulating.
        for question in questions:
            content = question.raw_entry.get("content")
            if isinstance(content, dict):
                content.pop("explanation", None)
                content.pop("referenced_table", None)

        explanation_map: Dict[int, str] = {}
        for match in matches:
            referenced_tables = extract_tables_for_match(
                args.pdf,
                match,
                index,
                table_cache,
                table_opts,
            )
            content = match.question.raw_entry.setdefault("content", {})
            if referenced_tables:
                content["referenced_table"] = referenced_tables
            skip_texts: List[str] = []
            for tbl in referenced_tables:
                for row in tbl.get("rows", []):
                    skip_texts.extend(str(cell) for cell in row if cell)
            explanation_text = extract_explanation_text(
                match,
                index,
                skip_texts=skip_texts,
            )
            explanation_map[match.question.number] = explanation_text
            content["explanation"] = explanation_text

        logging.info(
            "Extracted explanations for %s questions",
            len(explanation_map),
        )

        if args.mismatch_report:
            payload = [asdict(item) for item in mismatches]
            try:
                if args.mismatch_report.parent and not args.mismatch_report.parent.exists():
                    args.mismatch_report.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logging.error(
                    "Failed to create directories for mismatch report %s: %s",
                    args.mismatch_report,
                    exc,
                )
            try:
                args.mismatch_report.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                logging.info(
                    "Wrote mismatch diagnostics to %s (%s items)",
                    args.mismatch_report,
                    len(payload),
                )
            except OSError as exc:
                logging.error(
                    "Failed to write mismatch report to %s: %s",
                    args.mismatch_report,
                    exc,
                )

        if args.explanation_json:
            payload = [question.raw_entry for question in questions]
            try:
                serialized = json.dumps(payload, ensure_ascii=False, indent=2)
            except (TypeError, ValueError) as exc:
                logging.error(
                    "Failed to serialize explanation JSON for %s: %s",
                    args.explanation_json,
                    exc,
                )
            else:
                try:
                    if args.explanation_json.parent and not args.explanation_json.parent.exists():
                        args.explanation_json.parent.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    logging.error(
                        "Failed to create directories for explanation JSON %s: %s",
                        args.explanation_json,
                        exc,
                    )
                try:
                    args.explanation_json.write_text(
                        serialized,
                        encoding="utf-8",
                    )
                    logging.info(
                        "Wrote explanations to %s",
                        args.explanation_json,
                    )
                except OSError as exc:
                    logging.error(
                        "Failed to write explanation JSON to %s: %s",
                        args.explanation_json,
                        exc,
                    )

        if not matches:
            logging.error("No questions matched textual content in the PDF.")
            return 2

        annotate_pdf(
            doc,
            index,
            matches,
            padding=args.padding,
            stroke_width=args.stroke_width,
            label_font_size=args.label_font_size,
            label_prefix=args.label_prefix,
            text_offset=args.label_offset,
        )

        logging.info("Saving annotated PDF to %s", args.output)
        doc.save(args.output, garbage=4, deflate=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

