#!/usr/bin/env python3
"""Build a concise, auditable cover-letter package for journal submission.

The script creates a draft letter plus a brief/checklist. It uses provided
metadata first and extracts only light cues from the manuscript. Missing fields
are left as explicit placeholders so Codex or the author can fill them safely.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import textwrap
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


PLACEHOLDER = "[TO FILL]"


def today() -> str:
    return dt.date.today().isoformat()


def clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def read_docx(path: Path) -> str:
    parts: list[str] = []
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    for para in root.findall(".//w:p", ns):
        runs = [node.text or "" for node in para.findall(".//w:t", ns)]
        line = clean_space("".join(runs))
        if line:
            parts.append(line)
    return "\n".join(parts)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    if path.suffix.lower() == ".docx":
        try:
            return read_docx(path)
        except Exception:
            return ""
    return path.read_text(encoding="utf-8", errors="replace")


def infer_title(text: str) -> str:
    for raw in text.splitlines():
        line = clean_space(raw)
        if not line:
            continue
        if line.lower() in {"abstract", "summary", "introduction", "references"}:
            continue
        if 8 <= len(line) <= 180:
            return line
    return PLACEHOLDER


def infer_summary_hint(text: str, max_chars: int = 1200) -> str:
    lines = [clean_space(line) for line in text.splitlines() if clean_space(line)]
    for i, line in enumerate(lines):
        if line.lower() in {"abstract", "summary", "摘要"} and i + 1 < len(lines):
            return clean_space(lines[i + 1])[:max_chars]
        if line.lower().startswith(("abstract", "summary", "摘要")) and len(line) > 20:
            return clean_space(re.sub(r"^(abstract|summary|摘要)[:：]?", "", line, flags=re.I))[:max_chars]
    return clean_space(" ".join(lines[:8]))[:max_chars]


def split_pipe_record(raw: str, fields: list[str]) -> dict[str, str]:
    values = [part.strip() for part in raw.split("|")]
    values += [""] * (len(fields) - len(values))
    return dict(zip(fields, values[: len(fields)]))


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def val(value: str, fallback: str = PLACEHOLDER) -> str:
    return clean_space(value) if clean_space(value) else fallback


def trim_terminal_punctuation(text: str) -> str:
    return clean_space(text).rstrip(".。;；")


def paragraph(*sentences: str) -> str:
    return clean_space(" ".join(sentence for sentence in sentences if clean_space(sentence)))


def build_letter(args: argparse.Namespace, title: str, summary_hint: str) -> str:
    journal = val(args.target_journal, "[TARGET JOURNAL]")
    editor = val(args.editor, "Editor")
    article_type = val(args.article_type, "Review")
    main_finding = val(args.main_finding, "[one-sentence central finding or argument]")
    field_impact = val(args.field_impact, "[why this changes the field or review agenda]")
    broad_audience = val(args.broad_audience, "[why readers beyond the narrow subfield should care]")
    journal_fit = val(args.journal_fit, "[why this belongs in the target journal]")
    comparison = val(args.comparison, "[how this differs from existing reviews or alternative approaches]")
    related = val(args.related_manuscripts, "We are not aware of related manuscripts with overlapping authorship that are in press or under consideration elsewhere.")
    prior_discussion = val(args.prior_editor_discussion, "We have had no prior editorial discussion about this manuscript.")
    conflicts = val(args.conflicts, "The authors declare no competing interests relevant to this submission.")
    ethics = val(args.ethics, "")
    ai_disclosure = val(args.ai_disclosure, "")
    corresponding = val(args.corresponding_author, "[corresponding author name and email]")

    salutation = f"Dear {editor},"
    opening = paragraph(
        f"Please consider our manuscript, \"{title},\" as a {article_type} for {journal}.",
        f"The paper addresses {trim_terminal_punctuation(val(args.background_question, '[the field-level problem or question]'))}.",
    )
    pitch = paragraph(
        f"Its central message is: {trim_terminal_punctuation(main_finding)}.",
        f"The significance is that {trim_terminal_punctuation(field_impact)}.",
        f"This should interest a broad readership because {trim_terminal_punctuation(broad_audience)}.",
    )
    fit = paragraph(
        f"Compared with existing reviews or alternative approaches, {trim_terminal_punctuation(comparison)}.",
        f"The manuscript is a strong fit for {journal} because {trim_terminal_punctuation(journal_fit)}.",
    )
    confidential = paragraph(
        related,
        prior_discussion,
        conflicts,
        ethics,
        ai_disclosure,
        f"The manuscript is not published elsewhere and is not under consideration by another journal; all authors have approved this submission to {journal}.",
    )
    close = paragraph(
        f"Correspondence should be addressed to {corresponding}.",
        "Thank you for considering this manuscript.",
    )

    body = "\n\n".join(
        [
            args.date or today(),
            salutation,
            opening,
            pitch,
            fit,
            confidential,
            close,
            "Sincerely,\n[Corresponding author]",
        ]
    )
    if args.include_reviewer_section:
        body += "\n\nSuggested and excluded referees are listed in the accompanying table."
    if summary_hint and args.include_manuscript_hint:
        body += "\n\n<!-- Manuscript hint for drafting only; remove before submission:\n" + summary_hint + "\n-->"
    return body


def build_brief(args: argparse.Namespace, title: str, letter: str, missing: list[str]) -> str:
    return textwrap.dedent(
        f"""
        # Cover Letter Brief

        - Target journal: {val(args.target_journal, '[TARGET JOURNAL]')}
        - Article type: {val(args.article_type, 'Review')}
        - Title: {title}
        - Editor: {val(args.editor, 'Editor')}
        - Cover letter word count: {word_count(letter)}
        - Recommended ceiling: {args.max_words} words

        ## Editor Pitch

        - Main finding/argument: {val(args.main_finding)}
        - Field impact: {val(args.field_impact)}
        - Broad-audience appeal: {val(args.broad_audience)}
        - Journal fit: {val(args.journal_fit)}
        - Competitor/alternative comparison: {val(args.comparison)}

        ## Confidential/Submission Items

        - Related manuscripts: {val(args.related_manuscripts)}
        - Prior editor discussion: {val(args.prior_editor_discussion)}
        - Competing interests: {val(args.conflicts)}
        - Ethics/policy note: {val(args.ethics, '[none supplied]')}
        - AI-use disclosure: {val(args.ai_disclosure, '[none supplied]')}
        - Corresponding author: {val(args.corresponding_author)}

        ## Missing Fields

        {chr(10).join(f'- {item}' for item in missing) if missing else '- None'}
        """
    ).strip() + "\n"


def build_checklist(args: argparse.Namespace, letter: str) -> str:
    too_long = word_count(letter) > args.max_words
    return textwrap.dedent(
        f"""
        # Cover Letter Checklist

        - Keep it short enough for an editor's first-pass decision: {'NEEDS CUT' if too_long else 'OK'} ({word_count(letter)} words; ceiling {args.max_words}).
        - Re-state the manuscript's central finding or argument in one plain sentence.
        - Explain field-level impact and broad-audience interest, not only niche completeness.
        - Compare respectfully with existing reviews or alternative solutions; be explicit but not dismissive.
        - Do not repeat the abstract or introduction.
        - Declare related manuscripts in press, submitted, under appeal, or otherwise overlapping.
        - Declare prior editorial discussions if any.
        - Include required originality, author-approval, competing-interest, ethics, data/code, and AI-use statements as needed.
        - Suggest five or six independent referees when the journal permits it; avoid friends, close collaborators, same institution, recent coauthors, or conflicted experts.
        - Exclude referees only with concise, reasonable conflict or expertise-based justification.
        """
    ).strip() + "\n"


def write_reviewers(out_dir: Path, reviewers: list[str], excluded: list[str]) -> dict[str, str]:
    suggested_path = out_dir / "suggested_referees.csv"
    excluded_path = out_dir / "excluded_referees.csv"
    with suggested_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["name", "affiliation", "email", "expertise", "independence_rationale"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for raw in reviewers:
            writer.writerow(split_pipe_record(raw, fields))
    with excluded_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["name", "reason"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for raw in excluded:
            writer.writerow(split_pipe_record(raw, fields))
    return {"suggested_referees": str(suggested_path), "excluded_referees": str(excluded_path)}


def missing_fields(args: argparse.Namespace, title: str) -> list[str]:
    required = {
        "target_journal": args.target_journal,
        "title": title,
        "background_question": args.background_question,
        "main_finding": args.main_finding,
        "field_impact": args.field_impact,
        "broad_audience": args.broad_audience,
        "journal_fit": args.journal_fit,
        "comparison": args.comparison,
        "corresponding_author": args.corresponding_author,
    }
    return [name for name, value in required.items() if not clean_space(value)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a concise cover-letter draft, brief, reviewer tables, and checklist.")
    parser.add_argument("--manuscript", help="Optional manuscript path (.md/.txt/.docx) for title and summary hints.")
    parser.add_argument("--out-dir", default="./review-output/cover_letter")
    parser.add_argument("--target-journal", default="")
    parser.add_argument("--article-type", default="Review")
    parser.add_argument("--editor", default="")
    parser.add_argument("--date", default="")
    parser.add_argument("--title", default="")
    parser.add_argument("--background-question", default="")
    parser.add_argument("--main-finding", default="")
    parser.add_argument("--field-impact", default="")
    parser.add_argument("--broad-audience", default="")
    parser.add_argument("--journal-fit", default="")
    parser.add_argument("--comparison", default="")
    parser.add_argument("--related-manuscripts", default="")
    parser.add_argument("--prior-editor-discussion", default="")
    parser.add_argument("--conflicts", default="")
    parser.add_argument("--ethics", default="")
    parser.add_argument("--ai-disclosure", default="")
    parser.add_argument("--corresponding-author", default="")
    parser.add_argument("--reviewer", action="append", default=[], help="Repeatable: Name|Affiliation|Email|Expertise|Why independent.")
    parser.add_argument("--exclude-reviewer", action="append", default=[], help="Repeatable: Name|Reason.")
    parser.add_argument("--max-words", type=int, default=650)
    parser.add_argument("--include-reviewer-section", action="store_true")
    parser.add_argument("--include-manuscript-hint", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manuscript_text = read_text(Path(args.manuscript)) if args.manuscript else ""
    title = val(args.title, infer_title(manuscript_text))
    summary_hint = infer_summary_hint(manuscript_text)
    letter = build_letter(args, title, summary_hint)
    missing = missing_fields(args, title)

    letter_path = out_dir / "cover_letter_draft.md"
    brief_path = out_dir / "cover_letter_brief.md"
    checklist_path = out_dir / "cover_letter_checklist.md"
    summary_path = out_dir / "cover_letter_summary.json"

    letter_path.write_text(letter + "\n", encoding="utf-8")
    brief_path.write_text(build_brief(args, title, letter, missing), encoding="utf-8")
    checklist_path.write_text(build_checklist(args, letter), encoding="utf-8")
    reviewer_paths = write_reviewers(out_dir, args.reviewer, args.exclude_reviewer)

    summary = {
        "out_dir": str(out_dir),
        "cover_letter": str(letter_path),
        "brief": str(brief_path),
        "checklist": str(checklist_path),
        "word_count": word_count(letter),
        "max_words": args.max_words,
        "missing_fields": missing,
        **reviewer_paths,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
