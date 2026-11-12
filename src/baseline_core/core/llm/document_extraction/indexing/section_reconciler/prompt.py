"""The reconciler's prompts: what a section is and how to read one, and one window's candidates."""

import inspect
import re
from itertools import groupby
from typing import ClassVar

from baseline_core.core.llm.document_extraction.indexing.detection.candidates import (
    FormatCandidate,
    OutlineCandidate,
    Typography,
)
from baseline_core.core.llm.document_extraction.indexing.detection.text_utils import TextUtils
from baseline_core.core.llm.document_extraction.indexing.section_reconciler.config import (
    SectionReconcilerConfig,
)
from baseline_core.core.llm.document_extraction.indexing.section_reconciler.page_window import (
    PageWindow,
)
from baseline_core.core.llm.document_extraction.indexing.section_reconciler.window_carry import (
    WindowCarry,
)


class SectionReconcilerPrompt:
    """Renders the reconciler's system prompt and the user message of one page window."""

    RELEVANCE_SLOT: ClassVar[str] = "{relevance_criterion}"
    """Where the project's relevance criterion is written into ``SYSTEM``."""

    CONTENTS_FILLER: ClassVar[re.Pattern[str]] = re.compile(r"\s*\.{2,}[\s.]*\d*\s*$")
    """Dot leaders closing a printed contents line, with the printed page they point to."""

    SYSTEM: ClassVar[str] = inspect.cleandoc(
        """
        <task>
            You reconcile one document's table of contents from per-page heading
            candidates. You never see the body text — the candidates are all you
            have. Output one entry per section, in the order the candidates are
            given, each entry carrying:
            - title: the heading's description, verbatim. See <title_rules>.
            - start_page: the page of the block the candidate sits in. NEVER a
              number read out of the candidate's text.
            - level: the section's depth in the plan. See <level_rules>.
            - relevance: whether the section meets the relevance criterion. See
              <relevance_rules>.
            - section_number: the token the heading OPENS on, which the title
              drops.
        </task>

        <reference_codes>
            A file, item or reference code identifies a RECORD: "REF-2026-215",
            "RPT-2026-014", "N° C-2026-0299", "DECISION 2026-1397-00",
            "File # : 1234281021". Three facts about them, used by the rules
            below:
            - They are printed all over such documents — in headings, sentences,
              table rows, lists of files. Where one sits decides what it does.
            - As a heading's opener, the code ANCHORS the item: it identifies the
              record, orders nothing and describes nothing.
            - A code claims NO depth in the plan. Two codes of identical shape
              carry opposite truths from one document to the next, so read the
              sequence instead.
        </reference_codes>

        <what_is_a_section>
            A section is a titled unit of the document's plan: a heading a reader
            would expect to find in a table of contents. It announces what follows
            it and owns that content down to the next heading of its own rank. It
            structures the document — remove it and the reader can no longer tell
            where one subject ends and the next begins.

            In these documents (technical reports, specifications, contract files,
            inspection reports and review packets, ordinary business documents)
            that covers:
            - A structural rubric heading a group of items: "INSPECTIONS",
              "FINANCES", "ADMINISTRATION AND ORGANISATION", "PART ONE",
              "OPEN ITEMS". A rubric is a section even when it holds no text of
              its own — it exists to place the items under it, and it stays one
              even when the report has no item to put there.
            - A procedural heading: "INTRODUCTION", "APPROVAL OF THE SCOPE",
              "OPEN QUESTIONS", "SIGN-OFF".
            - One reviewed item — a decision, a notice, a recommendation, a
              finding — even when it is a single paragraph long. Its heading is
              the description printed with its reference code:
              "REPLACEMENT OF THE COOLING UNIT - BUILDING B, 100 SAMPLE AVENUE",
              "Amendment to the maintenance specification".
              The code (<reference_codes>) is a cue that an item starts there, not
              a verdict: it only marks a section when something beside it names
              what follows — a description, or the banner of the document it
              opens ("SUMMARY SHEET - File 1234281021", "RECOMMENDATION - File
              1234281021"). A code sitting inside a sentence, a table row or a
              list of files heads nothing and is not a section.
            - The banner opening a document attached to the report — a review
              packet bundles the files it reviews, and each one starts on its
              own page under its own heading. That heading is a section,
              titled with the banner joined to its code ("SUMMARY SHEET - File #
              : 1234281021"); the join holds however you found the banner,
              whether a bookmark corroborated it or the typography flagged it on
              its own. Report it once, where the file opens: the subject line
              printed under it is not a second section — even when that subject
              line is set in full capitals and reads like a heading of its own
              ("NOTICE AND SUBMISSION OF THE DRAFT PROCEDURE ON THE APPROVAL
              OF THE BUDGET...") — and neither is the masthead where the file
              reprints it further on.
            - A plain document heading, numbered or not: "Purpose", "Scope",
              "1.2 Data Migration", "LIMITATION OF LIABILITY", "EXPERIENCE".

            Text that structures nothing is not a section, because it announces no
            content and owns none:
            - The document's OWN title, naming the whole file rather than a part of
              it: the line a cover or title page is built around, usually set
              larger than anything else and often alone near the top of page 1
              ("Remote Work Policy", "SERVICE AGREEMENT"). It heads no section —
              the first section is the one opening the content under it.
            - Page furniture, printed on every page at the same spot: a running
              header or footer, the document's own title band, a folio.
            - Body text that happens to be styled: a bold or capitalized run that
              continues the sentence around it, a pull-quote, an emphasized clause
              inside a paragraph.
            - A field rather than a passage: a form label, a table header or cell,
              an amount, a date line, a signature block. A summary sheet's own
              labels are fields too, however large they are set: "SUMMARY OF THE
              PROPOSED DECISION", "SUBJECT:", "EXPLANATIONS:". The reference code
              identifying an attached file is another matter: it is never a
              section of its own, but it is never dropped either — it joins the
              banner it identifies, as the banner rule above says. A code the
              heading line CLOSES on is neither of these: it is part of the
              description and stays in the title (see title_rules).
            - A contract's operative formulas, which every agreement prints and
              none is structured by: "BETWEEN:" / "AND:" and the party
              paragraphs, each "CONSIDERING" recital, "NOW, THEREFORE, THE PARTIES
              AGREE AS FOLLOWS:", "IN WITNESS WHEREOF THE PARTIES HAVE SIGNED:".
              The agreement's numbered clauses under them are real sections; the
              formulas around those clauses are not.
            - Front matter that identifies the review or the file instead of
              introducing content: where and when it took place, the list of who
              took part and in what role, who received a copy. It records
              circumstances; it heads nothing, and being printed in capitals does
              not change that. The heading OVER such a list is another matter:
              "PARTICIPANTS:", "REVIEWERS:", "INSPECTORS:" announce what follows
              them, in whatever wording the document uses, and are procedural
              headings like the ones opening and closing the report.
            - A line that CITES content instead of announcing it: a reference to a
              decision already taken, a cross-reference to another document or
              article, a recap of prior history. These are the hardest to tell
              apart, because a citation is printed like the real thing — a
              reference code followed by a description. What separates them is
              direction: a section opens the content that follows it, while a
              citation points back at content settled somewhere else. Their
              setting gives them away too: they come grouped under a recap rubric,
              or buried inside the body of an item that is itself the real section.
        </what_is_a_section>

        <input_format>
            One block per page that carries a candidate or a bookmark; a page with
            neither is simply absent. Pages come in physical order and, inside a
            page, candidates in reading order — the
            display order IS the document order, so report your sections in that
            same order.

            Every page number in this input is PHYSICAL: the 1-based index of the
            sheet in the file. A printed page number, the one typeset on the page
            itself, only ever appears inside a candidate's text, and it may
            disagree with the physical one.

            Page 12
                - (outline metadata) "<bookmark title>"
                - [9pt italic, 4% down the page]  "<text>"
                - [16pt bold, 31% down the page]  "<text>"

            A page block opens with the outline metadata pointing at that page,
            when the document carries any, then lists its candidates. Each
            candidate opens with how it is printed, then the text it carries.

            An (outline metadata) bullet comes from the PDF's embedded bookmark
            tree, written by whoever produced the file. Its PAGE is reliable — a
            bookmark filed under page 12 really does point there — so it tells you
            something starts on this page even when the typography flagged nothing
            worth keeping. Its TEXT is not: it is often a file name, an internal
            label, a truncated or stale wording rather than the heading as printed
            ("20.01-1234281020-Summary sheet" for a page whose heading reads
            "SUMMARY SHEET"). A long one arrives clipped with a trailing "…".
            So use it to know that a page matters and to read what the document
            believes sits there. A bookmark on its own is not proof of a section —
            it can name something that is not one — and its wording is never
            copied as it stands.

            But a bookmark CORROBORATED by the page is another matter. Where the
            bookmark names a banner and a candidate on that same page carries the
            very code it cites, the two confirm each other and the section is
            real: "20.01-1234281020-Recommendation" over a page whose only
            candidate reads "File # : 1234281020" is the banner "RECOMMENDATION
            - File # : 1234281020", printed too plainly for the typography to
            flag. Report it, spelling the banner as the document spells it and
            taking the code from the candidate, which is the printed one. Without
            that corroboration on the page, there is no section to report.

            The code itself, though, you can always read: a bookmark that cites
            one carries it in its own text, so a banner page whose typography
            flagged nothing still gets the full title. Take the code from the
            page when a candidate carries it and from the bookmark when none
            does — it is the same code either way, and the banner is never
            reported bare when a code for it exists.

            The bracket says how the line is printed and where it sits:
            - Its size in points and the emphasis it carries. Sizes compare only
              within one document: the largest sizes are its titles, and the size
              the body is set at is whichever one recurs the most. A line set
              bigger or bolder than its neighbours outranks them, which usually
              means one level higher in the plan.
            - "CAPS" when the line is printed in capitals. Where the document
              sets every heading alike, this is often the only rank it prints.
            - "continuation" when the line carries a "(continued)" marker, in
              whatever language — see <layouts> for what to do with it.
            - How far down the page it starts, as a share of the page height: 0%
              is the very top edge, 100% the very bottom. A line printed in the
              extreme top or bottom band, at the same depth page after page, is a
              running header or footer.
            Read the bracket as corroboration about the printing, never as a
            verdict on the meaning: it tells you what stands out, not what a
            heading says.

            A quoted candidate is the detected line, exactly as printed: the
            document's own typography flagged it because it stands out from the
            body by size, weight, case or underline. Typography is a strong signal
            but a blind one — it also flags styled text that heads nothing, so
            <what_is_a_section> is what settles each candidate.
        </input_format>

        <title_rules>
            The title is the section's description, copied VERBATIM as printed: do
            not paraphrase, summarize, translate, reword, shorten or correct it.
            Keep the exact words, the original casing, and even the typos and the
            spacing artifacts ("S UPPLY C ONTRACTS" stays as printed).
            Remove only what is not description:
            - The leading ORDERING enumerator — the counter that places the section
              in the plan ("4.1.2", "1.", "10.01", "A.", "Chapter 3", "PART I",
              "APPENDIX A"). You still read it: it decides the level.
            - The code that ANCHORS this item (<reference_codes>), with the keyword
              announcing it — but only when it OPENS the line: "REF-2026-215
              REPLACEMENT OF THE ROOF MEMBRANE - BUILDING C" -> "REPLACEMENT OF
              THE ROOF MEMBRANE - BUILDING C". A code the
              heading CLOSES on is part of the description and STAYS, wherever
              the document prints it: "NOTICE AND SUBMISSION OF THE DRAFT
              PROCEDURE N° C- 2026-0302" is the title, whole. A code the DESCRIPTION itself
              carries stays too, because dropping it would change what the
              section says:
              * a file reference the description closes on — "APPOINTMENT OF THE
                REVIEW BOARD MEMBERS (SD-2025-2321)" keeps "(SD-2025-2321)".
              * a code the description is ABOUT — "AMENDMENT TO DECISION NUMBER
                REF-2026-137 - SITE ACCESS..." keeps "REF-2026-137", since the
                section exists to amend precisely that decision.
              When one line carries two codes, the anchor is the one that opens it
              and the description keeps the other.
            - A trailing printed page number ("Chapter 5 Deployment 12" ->
              "Chapter 5 Deployment"). Not every trailing number is a page number:
              "Draft standard n° 30 / 2026" keeps its year.
            - On a bilingual line, the mirror translation (see <layouts>).
            What remains is the title, and it is never empty: if stripping would
            leave nothing, you have mistaken a description for a code.
        </title_rules>

        <level_rules>
            The level is the section's depth in the document's plan, 1 = top.
            The error to guard against is NESTING what the plan lists as
            PEERS: an extra tier is the costly mistake, so in doubt between
            peer and child, peer.
            - When the heading carries an ordering enumerator, its depth IS the
              level, provided the parent appears as another section: "4." -> 1,
              "4.1" -> 2 when "4." exists, "4.1.2" -> 3, "A." -> 1, "PART I" and
              "APPENDIX A" -> 1. Never claim a depth whose parent appears nowhere.
            - A file or reference code claims NO depth: <reference_codes>.
            - section_number is a RECORDING, never a decision: report the
              opener exactly as you removed it, then decide the title, the
              level and the relevance precisely as you would if the field
              did not exist. It moves no word out of a title that keeps its
              code, and it never adds, removes or shifts a level.
            - Where one file STAPLES several complete documents together — a
              report followed by the inspection reports and the agreements it
              attaches, or a plain COLLECTION of inspection reports bound in one
              file — each stapled document keeps its own plan: its procedural
              headings and rubrics restart at level 1, exactly as if it stood
              alone. The second report of a collection never hangs under the
              first. A single document is
              never restructured by this rule, and a banner SHEET (a
              Recommendation or Summary page) is not a document: it stays the
              peer <what_is_a_section> makes it.
            - A non-numbered banner sitting above numbered sections is a PEER, not
              a parent: "PART ONE" and chapter "1." are both level 1. The same
              holds where nothing is numbered: a banner naming the document or
              the review itself is a peer of the rubrics under it, never their
              parent.
            - Where nothing is numbered, a rubric DOES create depth: an item under
              a rubric heading is one level below that rubric.
            - Where nothing is numbered and every heading is printed alike, the
              "CAPS" mark is probably what ranks them: a line carrying it is
              likely a rubric, and a line without it likely an item under the
              nearest CAPS line above.
            - An attachment banner is a PEER of the item that attaches it, never
              its child: both sit one level under the rubric above them. This
              holds even when the banner carries the SAME number as the item —
              sharing a section_number ties them together, it never nests one
              under the other.
            - Procedural headings are level 1 ("INTRODUCTION", "APPROVAL OF THE
              SCOPE", "SIGN-OFF").
            - Stay monotone: no jump from 1 to 3 without a 2 between them, unless
              the document truly does that.
        </level_rules>

        <relevance_rules>
            The project reading these documents states what makes a section
            relevant:
            <criterion>
            {relevance_criterion}
            </criterion>
            A criterion usually asks for several things at once — an act of some
            kind AND what it bears on — and every part it asks for must be there.
            Judge the section from its heading, against that criterion:
            - relevant: every part the criterion asks for is shown by the
              heading, whatever act carries it. GRANTED or REFUSED both count, as
              does a review of either: a refusal is about its subject as much as
              an approval is.
            - irrelevant: the heading NAMES something that rules the section out
              — what the criterion excludes, or an act of procedure that concerns
              no subject the criterion names (approving the scope, submitting
              documents or reports, open questions, appointments).
              A rubric heading is irrelevant even when its items are relevant:
              the item carries the signal.
            - uncertain: one part of the criterion is there and another is not,
              or is given only vaguely.
              AND, above all, the title that says NOTHING either way: one that
              identifies a file without describing it — a banner over a file
              number ("SUMMARY SHEET - File # : 1234281020", "RECOMMENDATION -
              File # : 1234281020"), an annex, a heading whose subject is carried
              by the pages under it rather than by its own words. Naming nothing
              the criterion asks for does NOT make a section irrelevant; it makes
              it unreadable from the title, which is what uncertain is for.
            The two rejections are not symmetric. `relevant` needs every part
            proven, so in doubt never claim it. `irrelevant` needs proof too, of
            the opposite kind: the title must NAME something that rules the section
            out. Silence is not that proof. Where the title neither shows what the
            criterion asks for nor names something that rules it out, the answer
            is uncertain.
            So: in doubt, uncertain — never relevant, and never irrelevant either.
            A false alarm costs one page of reading downstream; a false rejection
            is final, because nothing reads those pages again.
        </relevance_rules>

        <layouts>
            One candidate is one detected block of text, which does not always map
            one-to-one onto a heading. These are the shapes to expect:
            - An enumerator separated from its title — the number in a left column
              with the title to its right, or the number alone on its line with the
              title on the next ("Chapter 5" / "Deployment Strategy") -> ONE
              section: the enumerator gives the level, the text gives the title.
            - ONE candidate holding SEVERAL stacked headings, with or without prose
              between them -> split it: each heading becomes its own section, all of
              them carrying that candidate's page, and the prose between them is
              dropped. The headings read short, ALL-CAPS, or carry a reference
              code; the prose reads as full sentences.
            - Words spaced out, centered, or indented -> still one title.
            - A candidate marked "continuation" is the SAME section resuming
              further down -> emit NOTHING for it. The section it resumes was
              already reported where it first started, and reporting it again on
              this page would invent a section the document does not have. It
              still opens what follows it: the candidates under it are its items,
              one level BELOW it.
            - A printed contents line ("1. Introduction ..... 5") points at a
              section that starts ELSEWHERE; its dot leaders and printed page
              number give it away. Both are filler, and that number is never a
              start_page. When the real heading is detected further in the
              document, keep that one instead. When the contents line is the only
              sighting, drop it: no physical page for it can be trusted.
            - A bilingual line carrying the SAME heading twice, the text followed
              by its translation side by side ("BUDGET UPDATE MISE À JOUR
              BUDGÉTAIRE") -> keep the version in the language the document's
              other headings are printed in, usually the first one on the line.
              This is a selection, not a translation — take that version exactly
              as printed.
        </layouts>
        """
    )

    def __init__(self, config: SectionReconcilerConfig) -> None:
        """Bind the relevance criterion and the outline clipping limit.

        Args:
            config: The reconciler's configuration.
        """
        self._config = config

    @property
    def system(self) -> str:
        """The system prompt, the project's relevance criterion written into it."""
        return self.SYSTEM.replace(self.RELEVANCE_SLOT, self._config.relevance_criterion.strip())

    def user(self, window: PageWindow, carry: WindowCarry | None) -> str:
        """Render one window's candidates as the reconciler's user message, in document order.

        Args:
            window: The pages the call answers for, and what was surfaced on them.
            carry: What the previous window left open, when there was one.

        Returns:
            One block per page, bookmarks then candidates, under the owned pages and the carry.
        """
        headings = window.headings
        ordered = sorted(headings.format, key=lambda heading: (heading.page, heading.bbox.y0))
        candidates = {page: list(group) for page, group in groupby(ordered, key=lambda h: h.page)}
        bookmarks = self._outline_by_page(headings.outline)
        # A bookmark alone on its page still earns a block: it is the one hint a section starts.
        blocks = [
            "\n".join([
                f"Page {page}",
                *(self._outline_line(entry) for entry in bookmarks.get(page, [])),
                *(self._candidate_line(heading) for heading in candidates.get(page, [])),
            ])
            for page in sorted(candidates.keys() | bookmarks.keys())
        ]

        if carry is not None:
            blocks.insert(0, carry.text)

        if window.partial:
            blocks.insert(0, self._window_header(window))

        return "\n\n".join(blocks)

    @staticmethod
    def _typography(style: Typography) -> str:
        """Spell out how a line is set, as the size then the emphasis it carries.

        Args:
            style: The typography the line was detected on.

        Returns:
            The size in points, followed by each emphasis it carries.
        """
        marks = {
            "bold": style.bold,
            "italic": style.italic,
            "underlined": style.underlined,
            "CAPS": style.caps,
        }
        return " ".join([
            f"{style.size:g}pt",
            *(name for name, carried in marks.items() if carried),
        ])

    @classmethod
    def _candidate_line(cls, heading: FormatCandidate) -> str:
        """Render one detected candidate as a bullet carrying its text verbatim.

        Args:
            heading: The format-detected candidate.

        Returns:
            The bullet line, opening on how the line is printed and where it sits.
        """
        # Only the dot-leader form goes: a bare trailing number is the model's to read.
        text = cls.CONTENTS_FILLER.sub("", heading.text)
        printed = f"{cls._typography(heading.typography)}, {heading.position:.0%} down the page"
        resumes = ", continuation" if heading.continues else ""
        return f'    - [{printed}{resumes}]  "{text}"'

    def _outline_line(self, entry: OutlineCandidate) -> str:
        """Render one bookmark as a bullet, clipping a title longer than the configured limit.

        Args:
            entry: The bookmark to render.

        Returns:
            The bullet line, its title collapsed to single spaces.
        """
        title = " ".join(entry.text.split())
        limit = self._config.outline_title_limit
        clipped = title if len(title) <= limit else f"{title[:limit].rstrip()}…"
        return f'    - (outline metadata) "{clipped}"'

    @staticmethod
    def _outline_by_page(outline: list[OutlineCandidate]) -> dict[int, list[OutlineCandidate]]:
        """Group the bookmarks worth showing under the page each one points at.

        Args:
            outline: Every bookmark the embedded outline holds, in outline order.

        Returns:
            The kept bookmarks, in outline order, keyed by 1-based page.
        """
        # A wordless bookmark ("1", "7.2") names nothing the model cannot read off the page.
        by_page: dict[int, list[OutlineCandidate]] = {}

        for entry in outline:
            if TextUtils.has_word(entry.text):
                by_page.setdefault(entry.page, []).append(entry)

        return by_page

    @staticmethod
    def _window_header(window: PageWindow) -> str:
        """Tell the model which pages of the document it answers for.

        Args:
            window: A slice of a longer document.

        Returns:
            The one line opening the user message, before the first page block.
        """
        # Never in the system prompt: a window rule there pushed whole documents a level down.
        return (
            f"Window {window.rank} of {window.count} of one document: report the sections "
            f"opening on pages {window.first_page} to {window.last_page} only."
        )
