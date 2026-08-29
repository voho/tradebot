"""The ledger's own format, made CI-enforced instead of exhortative.

Section E exists because twenty verification passes wrote ~64 lines of prose
each at the top of section D and buried the backlog table (R-158). R-158's fix
was a five-cell table -- "that signal is one row wide". Prose came straight
back one level down: by 08-28 the numbered rows averaged 3,359 characters
against 119 for the first six ever written, a 28x inflation, and the seven
newest carried a *sixth* cell against a five-column header, which Markdown
drops silently on render.

Both defects propagate the same way -- each pass copies the row above it -- so
neither is fixable by asking future sessions to be brief. R-169 made them
mechanical instead. The numbers below are the enforcement.

The last test here is not about formatting: it is the one step-0 check the
routine calls higher-priority than the backlog, run in CI.
"""

import re
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[1] / "docs" / "LEDGER.md"
EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"

#: Section E's live rows may not exceed this per cell. The format as R-158
#: designed it ran 84-246 characters for a whole row, so this is ~5x looser
#: than the shape it protects and would never have bound a working pass.
CELL_CAP = 300

#: Sections A, C and D are registries, not one-row-per-pass logs, so they get
#: a looser cap than E -- but a cap, which is B-48/R-183's whole point. Before
#: it, C ran a 1,004-character median and a 3,234 maximum in the `why` column
#: and D a 759 median, in the two tables Step 0 sends every session to read.
#:
#: 500 is not a round number picked to bind gently. It is what the row needs
#: to keep doing its job: C's `why` column carries a "do not re-try X without
#: Y" ruling in 95 of 123 rows, median 281 characters, and its lede (the
#: configuration count) runs a 167 median -- so 500 holds a full ruling plus a
#: full lede with room to spare, and forces only the evidence recital out.
#:
#: Forcing that recital out is safe because R-183 measured where it already
#: lives: across the 111 C rows and 43 D rows carrying three or more distinct
#: numbers, a MEDIAN OF 100% of those numbers already appear in the section B
#: entry the row's own `ref` cell points at (>=80% on 108/111 and 42/43). The
#: registry prose was a paraphrase of the round, never a second source. That
#: is why the cap is a cap and not the content migration B-48 assumed it would
#: have to be -- there was nothing to migrate.
REGISTRY_CELL_CAP = 500

#: Which cap governs which section.
CAPS = {"A": REGISTRY_CELL_CAP, "C": REGISTRY_CELL_CAP,
        "D": REGISTRY_CELL_CAP, "E": CELL_CAP}

#: A row of a Markdown table splits on unescaped pipes only.
_PIPE = re.compile(r"(?<!\\)\|")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in _PIPE.split(line.rstrip("\n"))[1:-1]]


def _is_rule(cell: str) -> bool:
    return set(cell) <= set("-: ") and bool(cell)


def _table(section: str) -> tuple[list[str], list[tuple[int, list[str]]]]:
    """(header cells, [(line number, row cells)]) of a registry table.

    Section E's table ends at its first `###` subsection -- E-verbose holds
    the pre-cap originals verbatim and E-archive the pre-table prose, and
    neither is a live row. Fenced blocks are skipped for the same reason:
    the rows quoted inside E-verbose are evidence, not table rows.
    """
    lines = LEDGER.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(f"## {section}."))
    end = next((i for i, l in enumerate(lines[start + 1:], start + 1)
                if re.match(r"^## [A-Z]\.", l) or l.startswith("### ")), len(lines))

    header, rows, fenced, prev = None, [], False, None
    for i in range(start, end):
        line = lines[i]
        if line.startswith("```"):
            fenced = not fenced
            continue
        if fenced or not line.startswith("|"):
            continue
        cells = _cells(line)
        if _is_rule(cells[0]):
            header = prev  # the separator confirms the line above was the header
            continue
        if header is None:
            prev = cells
            continue
        rows.append((i + 1, cells))

    assert header, f"section {section} lost its table header"
    assert rows, f"section {section} lost its rows"
    return header, rows


def _section_e_rows() -> tuple[list[str], list[list[str]]]:
    header, rows = _table("E")
    return header, [c for _, c in rows]


def test_registry_tables_match_their_header_width():
    """A surplus cell is dropped silently by every Markdown renderer.

    Two ways in, both found on 08-28. Seven consecutive section E passes
    shipped a sixth 'full detail: none (...)' cell into a five-column table.
    And three rows in sections C and D wrote maths in prose -- `|Δf|`,
    `|Δleg|`, `|exposure*lambda*next_day_return|` -- whose absolute-value
    bars are cell separators, splitting one row into five or nine. That is
    the `|basis|` shift that ended section B's life as a table (R-41, R-44),
    still live years later: B-46's row rendered as far as "gate the combined
    re-target once, either on gross leg turnover (`Σ" and dropped R-154's
    actual measured conclusion, in the backlog table Step 0 sends every
    session to read. Escape the bar: `\\|`.
    """
    for section in ("A", "C", "D", "E"):
        header, rows = _table(section)
        for lineno, cells in rows:
            assert len(cells) == len(header), (
                f"docs/LEDGER.md:{lineno}: section {section} row "
                f"{cells[0][:40]!r} has {len(cells)} cells against a "
                f"{len(header)}-column header. The surplus is invisible when "
                "rendered -- usually an unescaped `|` in maths or notation. "
                "Write it `\\|`.")


def test_table_cells_stay_within_their_section_cap():
    """No table cell in this file may hold a paragraph.

    Section E has been capped since R-169; R-183 extended the same rule to the
    registries after B-48. The failure both caps exist to stop is identical
    and spreads identically -- each new row is written by copying the row
    above it, so neither is fixable by asking future sessions to be brief.

    When this fires, the fix is never to squeeze the prose into fewer words.
    It is to put the paragraph where the file already says it goes: in the
    round's own section B entry, leaving the cell the ruling and the `ref`.
    """
    for section, cap in CAPS.items():
        header, rows = _table(section)
        for lineno, cells in rows:
            for name, cell in zip(header, cells):
                assert len(cell) <= cap, (
                    f"docs/LEDGER.md:{lineno}: section {section} row "
                    f"{cells[0][:40]!r}, column {name!r} is {len(cell)} "
                    f"characters against a {cap} cap. The paragraph belongs "
                    "in the round's section B entry; the cell keeps the "
                    "ruling and the ref. See R-169 (section E) and R-183 "
                    "(sections A, C, D).")


def test_capped_sections_keep_a_verbatim_archive():
    """Nothing is deleted -- the pre-cap rows stay, verbatim, below the table.

    R-169 established the shape (`### E-verbose`) and R-183 reused it for the
    registries. The archive is what makes the cap a trim rather than a loss,
    and what lets a capped cell end in a bare `...` without that ellipsis
    hiding anything.
    """
    text = LEDGER.read_text()
    for section in CAPS:
        assert re.search(rf"^### {section}-verbose\b", text, re.M), (
            f"section {section} is capped but has no `### {section}-verbose` "
            "archive. A cap without one deletes evidence, which this file "
            "does not do.")


def test_section_e_pass_numbers_descend_within_the_live_block():
    """ROUTINE.md Step 0b counts null passes as the numbered rows above the
    first `—`, and tiers its behaviour on that count. The count is only
    meaningful if the block is ordered and its top row states its height.
    """
    _, rows = _section_e_rows()

    block: list[int] = []
    for cells in rows:
        if not cells[0].isdigit():
            break
        block.append(int(cells[0]))

    if not block:
        return  # the newest row is a dispatched round: the count is 0

    assert block == sorted(block, reverse=True), (
        f"section E's live pass numbers are out of order: {block}. The "
        "section is newest-first everywhere.")
    assert block[0] == len(block), (
        f"the newest section E row says pass {block[0]} but {len(block)} "
        "numbered rows sit above the first `—`. Step 0b's counter reads the "
        "row count, so the two must agree.")


def test_no_undispatched_frozen_pre_registration():
    """ROUTINE.md Step 0's first check, run in CI.

    A frozen `r<nn>_shared.py` with no section B entry is work another
    session left in flight, and the routine ranks executing it above both
    the backlog and any new idea. The manual command for this was
    `ls experiments/*_shared.py | tail -3`, which sorts lexicographically:
    from 08-20 to 08-28 it reported `r99_shared.py` as the newest file while
    `r168_shared.py` sat on disk, and it failed in the direction that looks
    like success. Sorting numerically is the whole fix.
    """
    numbered = sorted(
        (int(m.group(1)), p)
        for p in EXPERIMENTS.glob("r*_shared.py")
        if (m := re.fullmatch(r"r(\d+)_shared\.py", p.name)))
    if not numbered:
        return

    text = LEDGER.read_text()
    for n, path in numbered:
        recorded = (re.search(rf"^### R-{n} ", text, re.M)
                    or re.search(rf"^.*IN PROGRESS: R-{n}\b", text, re.M))
        assert recorded, (
            f"{path.name} is a frozen pre-registration with no R-{n} entry in "
            "section B and no `IN PROGRESS: R-{n}` stub. Either it is in "
            "flight -- announce it with the stub, per R-131/R-133 -- or it "
            "was frozen and never dispatched, in which case executing it "
            "verbatim outranks anything else on the backlog.")
