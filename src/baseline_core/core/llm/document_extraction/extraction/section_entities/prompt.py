"""The text agent's prompts: what to read off one section, and what to leave out."""

import inspect
from typing import ClassVar

from baseline_core.core.llm.document_extraction.extraction.mention_harvester import Candidate
from baseline_core.core.llm.document_extraction.section import LocatedSection


class SectionEntitiesPrompt:
    """Renders the static system prompt of a project and the user prompt of one section."""

    # The system prompt stays static so the server caches it; the harvest goes in the user prompt.

    TASK: ClassVar[str] = inspect.cleandoc(
        """
        <task>
            You extract what ONE section of a document names. You receive its heading and
            its body text — never the rest of the document, and never its neighbours.
            Report only what THIS section says. The description of each field of the output
            schema is the project's instruction for that field: follow it.
            A section that names nothing yields an empty object, and for a purely
            administrative one (a table of contents, a distribution list, a signature block)
            that is the correct answer.
        </task>
        """
    )

    INPUT_FORMAT: ClassVar[str] = inspect.cleandoc(
        """
        <input_format>
            <opener>, when present, holds what the printed heading opens on before its
            description, verbatim: an ordering number ("4.1.", "PART II"), which names
            nothing, or the code the item is filed under, which this section owns.
            <section_title> holds the heading's description as printed, verbatim.
            <section_body> holds the text under it, read from its pages.

            The heading usually carries the anchoring code and the subject; the body carries
            the details, the parties and the references. Read both — many sections name
            their own code or their site only in the body.
        </input_format>
        """
    )

    RULES: ClassVar[str] = inspect.cleandoc(
        """
        <verbatim_values>
            Every mention's value is copied exactly as the section prints it: never
            completed, reformatted, translated, or rebuilt from another section. A value
            printed in two forms is reported in the form this section prints first.
        </verbatim_values>

        <own_versus_cited>
            Which role a mention takes is the one judgment no field can make alone, and a
            single line often feeds both: "Amendment 3 (ref. 2026-035) to the supply
            agreement 2021-126" — 2026-035 is what this section ISSUES, so it is 'own';
            2021-126 is what it POINTS AT, so it is 'cited'. Decide by direction: a section
            owns the values it IS or acts on (its own number, the site its work is on), and
            cites values settled in another document or section. In doubt, 'cited'.
        </own_versus_cited>

        <examples_are_illustrations>
            Every code, number or name quoted in these instructions and in the schema's field
            descriptions illustrates a SHAPE. Output a value only when it is printed in THIS
            section; a code you cannot point to in the section's own text is not in it.
        </examples_are_illustrations>

        <omission_rules>
            An OPTIONAL field with nothing to report is OMITTED — never null, never an
            empty list. A REQUIRED field is always answered, and null is a real answer
            there: it says the section prints nothing, which is what the schema asked.
            Never invent a code, a name or a place the section does not print, and leave
            out a mention you are unsure you read correctly rather than guess: a wrong code
            silently joins two unrelated matters.
        </omission_rules>
        """
    )

    CANDIDATE_CODES: ClassVar[str] = inspect.cleandoc(
        """
        <candidate_codes>
            The message may end on a <candidate_codes> list: values a program found printed
            in this section, each with its kind. That list concerns the ``mentions`` of the
            kinds it shows ALONE. Take your values of those kinds from it — role 'own' when
            this section owns the value, 'cited' when it merely refers to it — keep each kind
            as given, and add no value of those kinds that the list does not hold. Every
            other field, and the mentions of every other kind, you read from the section
            exactly as you would without the list.
        </candidate_codes>
        """
    )

    @classmethod
    def system(cls, kinds: dict[str, str]) -> str:
        """Render the system prompt of a project, the same for every section.

        Args:
            kinds: Each mention kind of the project, with a one-line description.

        Returns:
            The system prompt.
        """
        return "\n\n".join([
            cls.TASK,
            cls.INPUT_FORMAT,
            cls.kinds_block(kinds),
            cls.RULES,
            cls.CANDIDATE_CODES,
        ])

    @staticmethod
    def kinds_block(kinds: dict[str, str]) -> str:
        """Render the project's mention kinds, the only ones a mention may take.

        Args:
            kinds: Each mention kind of the project, with a one-line description.

        Returns:
            The block, one kind per line.
        """
        listed = "\n".join(f"    - {kind}: {description}" for kind, description in kinds.items())
        return (
            "<mention_kinds>\n"
            "    ``mentions`` takes every value the section prints that names a thing of one of\n"
            "    these kinds, and of no other kind:\n"
            f"{listed}\n"
            "</mention_kinds>"
        )

    @staticmethod
    def user(section: LocatedSection, candidates: list[Candidate], body_char_limit: int) -> str:
        """Render the opener, the title, the capped body, then the harvested candidates.

        Args:
            section: The section to read.
            candidates: The harvested values, in printed order, already capped.
            body_char_limit: Characters kept of the body.

        Returns:
            The user prompt.
        """
        opener = f"<opener>{section.section_number}</opener>\n\n" if section.section_number else ""
        prompt = (
            f"{opener}<section_title>\n{section.title}\n</section_title>\n\n"
            f"<section_body>\n{section.body[:body_char_limit]}\n</section_body>"
        )

        # The list's scope lives in the system prompt: announced here, it emptied other fields.
        if candidates:
            listed = "\n".join(
                f"{candidate.value}\t({candidate.kind})" for candidate in candidates
            )
            prompt += f"\n\n<candidate_codes>\n{listed}\n</candidate_codes>"

        return prompt
