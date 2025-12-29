#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Stitch Markdown files by glob mask into a single `All.md`-style aggregate.

This tool is designed for the canonical theory sources under
`docs/theory/files/` and supports both legacy 2-digit ids (`%%7d-00%%` …
`%%7d-99%%`) and 3-digit ids (`%%7d-100%%` …).

ВАЖНО:
- Сам выходной файл (--out) исключается из входного набора.
- Перед соединением у КАЖДОГО источника удаляется ЛЮБОЙ ВЕДУЩИЙ маркерный блок:
    [опц. '---']  %%7d-NN%%|%%7d-NNN%%  [опц. '---'] (+ пустые строки)
  чтобы в агрегате был ровно один заголовок блока (добавляем наш).

Example:
    Assemble theory canon from `docs/theory/files/` into `docs/theory/All.md`:

        python3 docs/theory/files/stitch_by_mask.py \\
            --dir docs/theory/files \\
            --mask "7d-*.md" \\
            --out docs/theory/All.md \\
            --order num \\
            --dedupe off

    Assemble into a chain of files with max 10MB per file (without splitting blocks):

        python3 docs/theory/files/stitch_by_mask.py \\
            --dir docs/theory/files \\
            --mask "7d-*.md" \\
            --out docs/theory/All.chain.md \\
            --chain \\
            --max-bytes 10000000 \\
            --order num \\
            --dedupe off
"""

import argparse
from pathlib import Path
import re
import sys
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Iterable, List, Optional, Set, Tuple

MARKER_RX_NAME = re.compile(r"(7d-\d{2,3}(?:-\d+)*(?:-[A-Za-z]+)?)")
MARKER_RX_TEXT = re.compile(r"%%\s*(7d-[^%\s]+)\s*%%")
BLOCK_HEADER_RX = re.compile(r"(?m)^---\s*$\n^%%\s*(7d-[^%\s]+)\s*%%\s*$\n^---\s*$")
LEADING_BLOCK_RX = re.compile(
    r"""
    ^\ufeff?\s*
    (?:---\s*\n)?
    \s*%%\s*(7d-[^%\s]+)\s*%%\s*\n
    (?:---\s*\n)?
    \s*\n*
    """,
    flags=re.UNICODE | re.VERBOSE,
)


@dataclass(frozen=True)
class SkipInfo:
    path: Path
    reason: str


@dataclass(frozen=True)
class SegmentPlacement:
    seg_id: str
    file: str
    start_line: int
    end_line: int
    size_bytes: int


def extract_id_from_name(p: Path) -> Optional[str]:
    m = MARKER_RX_NAME.search(p.name)
    if m:
        return m.group(1)
    return None


def find_all_markers(text: str) -> Set[str]:
    return {m.strip() for m in MARKER_RX_TEXT.findall(text)}


def find_all_block_headers(text: str) -> Set[str]:
    """
    Find embedded canonical block headers of the form:
        ---
        %%7d-...%%
        ---

    This intentionally ignores inline references like "см. %%7d-105%%".
    """
    return {m.strip() for m in BLOCK_HEADER_RX.findall(text)}


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def read_utf8(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="utf-8-sig")


def strip_leading_marker_block(text: str) -> Tuple[str, Optional[str]]:
    """
    Снимает ведущий блок-маркер (если он присутствует) и возвращает:
    (текст_без_блока, id_из_блока_или_None)
    Блок формы:
      ---
      %%7d-NN%% или %%7d-NNN%%
      ---
      \n (пустые строки)
    Все части (---) опциональны.
    """
    m = LEADING_BLOCK_RX.match(text)
    if not m:
        return text.lstrip("\ufeff"), None
    xx = m.group(1)
    end_pos = m.end()
    return text[end_pos:].lstrip("\n"), xx


def _utf8_len(s: str) -> int:
    return len(s.encode("utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _iter_candidate_files(
    root: Path,
    mask: str,
    recursive: bool,
) -> Iterable[Path]:
    """
    Yield files matching mask. Also tries a case-variant for .md/.MD.
    """
    masks: List[str] = [mask]
    if mask.endswith(".md"):
        masks.append(mask[:-3] + ".MD")
    elif mask.endswith(".MD"):
        masks.append(mask[:-3] + ".md")

    if recursive:
        # rglob is case-sensitive; we therefore apply fnmatch by ourselves.
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            for m in masks:
                if fnmatch(p.name, m):
                    yield p
                    break
    else:
        # glob is also case-sensitive; we apply it for each mask.
        for m in masks:
            yield from root.glob(m)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Сшивает файлы по маске в агрегат с едиными блок-маркерами "
            "%%7d-NN%%/%%7d-NNN%%."
        )
    )
    ap.add_argument(
        "--dir",
        default=".",
        help="Каталог поиска файлов (по умолчанию — текущий).",
    )
    ap.add_argument(
        "--mask", default="7d-*.md", help='Глоб-маска файлов, напр. "7d-*.md".'
    )
    ap.add_argument("--out", required=True, help="Путь выходного файла.")
    ap.add_argument(
        "--chain",
        action="store_true",
        help=(
            "Писать не один файл, а цепочку файлов по лимиту --max-bytes. "
            "Файл --out используется как манифест (список частей)."
        ),
    )
    ap.add_argument(
        "--max-bytes",
        type=int,
        default=10_000_000,
        help="Лимит размера одного выходного файла в байтах (по умолчанию 10_000_000).",
    )
    ap.add_argument(
        "--order",
        choices=["num", "name"],
        default="num",
        help="Порядок: num — по XX из имени 7d-XX, name — лексикографически.",
    )
    ap.add_argument(
        "--dedupe",
        choices=["off", "prefer-leaf", "prefer-aggregator"],
        default="off",
        help="Дедупликация агрегаторов/листовых (по умолчанию off).",
    )
    ap.add_argument(
        "--skip-empty", action="store_true", help="Пропускать пустые файлы."
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Искать файлы рекурсивно в подкаталогах --dir.",
    )
    args = ap.parse_args()

    root = Path(args.dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    skipped: List[SkipInfo] = []

    files = list(_iter_candidate_files(root, args.mask, args.recursive))
    # ❶ Исключаем сам выходной файл
    files2: List[Path] = []
    for p in files:
        if p.resolve() == out_path:
            skipped.append(SkipInfo(p, "skip: output file"))
            continue
        files2.append(p)
    files = files2

    if not files:
        print(
            f"[ERR] По маске ничего не найдено: {root}/{args.mask}",
            file=sys.stderr,
        )
        for s in skipped:
            print(f"[SKIP] {s.path} :: {s.reason}", file=sys.stderr)
        sys.exit(2)

    def sort_key(p: Path) -> Tuple[int, str]:
        if args.order == "name":
            return (0, p.name)
        m = MARKER_RX_NAME.search(p.name)
        if m:
            try:
                return (int(m.group(1)), p.name)
            except ValueError:
                return (10**9, p.name)
        return (10**9, p.name)

    files.sort(key=sort_key)

    # Предскан: определим агрегаторы (содержат чужие %%7d-YY%% внутри,
    # помимо собственного)
    file_info: List[Tuple[Path, Optional[str], str, Set[str], bool]] = []
    for p in files:
        text0 = normalize_newlines(read_utf8(p))
        if args.skip_empty and not text0.strip():
            skipped.append(SkipInfo(p, "skip: empty file"))
            continue
        own_id = extract_id_from_name(p)
        text_body, leading_id = strip_leading_marker_block(text0)
        block_id = own_id or leading_id

        if not block_id:
            cannot_infer_reason = (
                "skip: cannot infer id " + "(no 7d-* in name, no %%7d-*%% markers)"
            )
            skipped.append(SkipInfo(p, cannot_infer_reason))
            continue

        embedded_ids = find_all_block_headers(text_body)
        is_aggregator = bool(embedded_ids)
        file_info.append((p, block_id, text0, embedded_ids, is_aggregator))

    # Дедупликация
    selected: List[Tuple[Path, Optional[str], str, Set[str], bool]] = []
    if args.dedupe == "off":
        selected = file_info
    elif args.dedupe == "prefer-leaf":
        for tpl in file_info:
            if tpl[4]:
                skipped.append(
                    SkipInfo(
                        tpl[0],
                        "skip: aggregator (prefer-leaf)",
                    )
                )
                continue
            selected.append(tpl)
    elif args.dedupe == "prefer-aggregator":
        aggregators = [tpl for tpl in file_info if tpl[4]]
        leaves = [tpl for tpl in file_info if not tpl[4]]
        emitted_ids: Set[str] = set()
        for tpl in aggregators:
            selected.append(tpl)
            emitted_ids.update(tpl[3])
        for p, own_id, text0, markers, is_agg in leaves:
            if own_id and own_id in emitted_ids:
                skipped.append(
                    SkipInfo(
                        p,
                        "skip: leaf shadowed by aggregator id=" + str(own_id),
                    )
                )
                continue
            selected.append((p, own_id, text0, markers, is_agg))
    else:
        print(
            f"[ERR] Неизвестная стратегия дедупликации: {args.dedupe}",
            file=sys.stderr,
        )
        sys.exit(3)

    # Сборка с принудительным снятием ведущего маркера и добавлением нашего
    parts = []
    for p, own_id, text0, markers, is_agg in selected:
        text_stripped, xx = strip_leading_marker_block(text0)
        # id для шапки блока:
        # - из имени файла;
        # - если его нет — из снятого маркера;
        # - иначе — basename
        use_id = own_id or xx
        marker_tag = f"%%{use_id}%%" if use_id else f"%%{p.stem}%%"
        header = f"---\n{marker_tag}\n---\n\n"
        block = header + text_stripped
        if not block.endswith("\n"):
            block += "\n"
        parts.append((str(use_id) if use_id else p.stem, block.rstrip() + "\n"))

    if not args.chain:
        joined = "\n".join(block for _, block in parts)
        _write_text(out_path, joined)
        print(f"[OK] Собрано {len(parts)} блок(ов) → {out_path}")
    else:
        if args.max_bytes <= 0:
            print("ERROR: --max-bytes must be > 0", file=sys.stderr)
            sys.exit(2)

        base_dir = out_path.parent
        base_stem = out_path.stem

        part_idx = 1
        cur_lines: List[str] = []
        cur_bytes = 0
        cur_line_no = 1

        manifest_lines: List[str] = []
        placements: List[SegmentPlacement] = []

        def flush_part() -> None:
            nonlocal part_idx, cur_lines, cur_bytes, cur_line_no
            if not cur_lines:
                return
            part_name = f"{base_stem}.part{part_idx:03d}.md"
            part_path = base_dir / part_name
            _write_text(part_path, "".join(cur_lines))
            manifest_lines.append(part_name)
            part_idx += 1
            cur_lines = []
            cur_bytes = 0
            cur_line_no = 1

        for seg_id, block in parts:
            block_bytes = _utf8_len(block)
            block_lines = block.count("\n")

            if cur_lines and (cur_bytes + block_bytes) > args.max_bytes:
                flush_part()

            if not cur_lines and block_bytes > args.max_bytes:
                print(
                    "[WARN] Single block exceeds --max-bytes; writing it alone: "
                    f"id={seg_id} bytes={block_bytes} limit={args.max_bytes}",
                    file=sys.stderr,
                )

            start_line = cur_line_no
            cur_lines.append(block)
            cur_bytes += block_bytes
            cur_line_no += block_lines
            end_line = cur_line_no - 1

            placements.append(
                SegmentPlacement(
                    seg_id=seg_id,
                    file=f"{base_stem}.part{part_idx:03d}.md",
                    start_line=start_line,
                    end_line=end_line,
                    size_bytes=block_bytes,
                )
            )

        flush_part()

        _write_text(out_path, "\n".join(manifest_lines) + "\n")
        print(f"[OK] Собрано {len(parts)} блок(ов) → {len(manifest_lines)} file(s)")
        print(f"[OK] Manifest → {out_path}")

        map_path = base_dir / f"{base_stem}.parts_map.tsv"
        map_lines = ["seg_id\tfile\tstart_line\tend_line\tsize_bytes\n"]
        for pl in placements:
            map_lines.append(
                f"{pl.seg_id}\t{pl.file}\t{pl.start_line}\t{pl.end_line}\t"
                f"{pl.size_bytes}\n"
            )
        _write_text(map_path, "".join(map_lines))
        print(f"[OK] Parts map → {map_path}")

    info_out_excluded = (
        "[INFO] Сам выходной файл исключён из " + "входного набора: " + out_path.name
    )
    print(info_out_excluded)
    print(
        "[INFO] Ведущие маркеры исходников удалены и заменены единым "
        "заголовком каждого блока."
    )
    if skipped:
        print(f"[INFO] Пропущено файлов: {len(skipped)}", file=sys.stderr)
        for s in skipped:
            print(f"[SKIP] {s.path} :: {s.reason}", file=sys.stderr)


if __name__ == "__main__":
    main()
