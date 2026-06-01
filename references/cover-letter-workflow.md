# Cover Letter Workflow

Use this reference when preparing a cover letter, submission letter, presubmission pitch, or revision cover letter for a top-journal review manuscript.

## Quick Navigation

- Editorial premise: short editor-facing pitch, not a second abstract.
- Cover-letter shape: main finding, broad impact, positioning, related manuscripts, referees.
- Review-specific pitch: explain why this synthesis changes the field's map.
- Audit: check brevity, fit, competitor positioning, and disclosure completeness.
- Scripted drafting: use `cover_letter_builder.py` when manuscript and journal are known.

## Editorial Premise

Treat the cover letter as a short private conversation with the editor. It should help the editor decide whether the manuscript is important, timely, and appropriate for the journal. It is not a rewritten abstract.

For Nature-style submissions, combine three layers:

- Editorial pitch: main finding or argument, field-level impact, and broad readership.
- Fit and positioning: why this journal, why now, and how this manuscript differs from existing reviews or alternative approaches.
- Confidential logistics: related manuscripts, prior editor discussions, competing interests, suggested referees, excluded referees, and journal-specific declarations.

Useful public guidance:

- Nature Support, cover letter checklist: https://support.nature.com/en/support/solutions/articles/6000245674-cover-letter
- Nature initial submission guidance, cover letter and related manuscript sections: https://www.nature.com/nature/for-authors/initial-submission
- Nature Computational Science editorial on cover-letter pitfalls: https://www.nature.com/articles/s43588-022-00348-4

## Short Cover Letter Shape

Recommended maximum for first submission: 400-650 words unless the journal asks otherwise.

1. Header and salutation:
   - Date
   - Target journal
   - Editor name if known

2. Opening:
   - Manuscript title
   - Article type
   - Field-level question or problem

3. Core pitch:
   - One-sentence main finding, thesis, or organizing argument
   - Why the manuscript matters for the field
   - Why a broad audience should care

4. Positioning:
   - What existing reviews, frameworks, tools, or alternatives do not yet solve
   - How this manuscript advances beyond them
   - Keep comparisons explicit but respectful; do not attack competing work.

5. Journal fit:
   - Align the paper with the journal's readership and scope.
   - Explain the manuscript's editorial value, not only its completeness.

6. Confidential statements:
   - Related manuscripts in press, submitted, under appeal, or overlapping.
   - Prior discussions with editors.
   - Originality and author-approval declaration.
   - Competing interests, ethics, data/code availability, and AI-use disclosure if needed.

7. Referee suggestions:
   - Suggest five or six independent referees when appropriate.
   - Include name, affiliation, email, expertise, and independence rationale.
   - Avoid close collaborators, recent coauthors, same-institution researchers, mentors/mentees, and people with direct conflicts.
   - Exclude referees only with concise, reasonable justification.

## Review-Article Specific Pitch

For a top-journal review, the cover letter must explain why the paper is more than a summary.

Strong pitch patterns:

- This review reframes a fragmented field through a new organizing lens.
- This review resolves a conflict between technical progress and clinical/regulatory readiness.
- This review converts scattered evidence into a reusable taxonomy, roadmap, or reporting standard.
- This review identifies where hype exceeds evidence and proposes a disciplined agenda.

Weak pitch patterns:

- "This topic is important and timely" without a specific contribution.
- "We summarize recent advances" without a new synthesis.
- "This is the first comprehensive review" without proof and without explaining why that matters.
- Heavy self-promotion, endorsements, or inflated claims.

## Cover Letter Audit

Before submission, check:

- Can the editor understand the manuscript's single strongest argument in one minute?
- Does the letter explain impact for the journal's readers, not only for the authors' subfield?
- Does the comparison with alternatives clarify contribution without disparaging competitors?
- Are all related manuscripts disclosed?
- Are suggested reviewers credible, independent, and diverse in expertise/geography when possible?
- Are excluded reviewers justified by conflicts or expertise issues rather than personal preference?
- Does the letter avoid repeating the abstract, introduction, and author biography?
- Are journal name, editor name, manuscript title, article type, and declarations correct?

## Scripted Drafting

Use the bundled `cover_letter_builder.py` to create a structured package:

```powershell
$SKILL_DIR = "<skill-dir>"
python "$SKILL_DIR/scripts/cover_letter_builder.py" \
  --manuscript <manuscript-path> \
  --target-journal "<target-journal>" \
  --article-type "Review" \
  --background-question "<one-sentence field problem>" \
  --main-finding "<main thesis or organizing insight>" \
  --field-impact "<why this matters for the field>" \
  --broad-audience "<communities that will care>" \
  --journal-fit "<why this journal's readers need this synthesis>" \
  --comparison "<respectful comparison with competing reviews or alternatives>" \
  --corresponding-author "Name <email@example.com>" \
  --out-dir ./review-output/cover_letter
```

Outputs:

- `cover_letter_draft.md`
- `cover_letter_brief.md`
- `cover_letter_checklist.md`
- `suggested_referees.csv`
- `excluded_referees.csv`
- `cover_letter_summary.json`

Do not submit the generated draft directly. Treat it as a first pass for Codex and the human author to revise after journal-specific instructions are checked.
