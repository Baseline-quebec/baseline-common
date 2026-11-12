"""Injectable thresholds the HeadingFinder reads every document with."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class TextConfig(BaseModel):
    """Injectable thresholds for the all-caps heuristic."""

    model_config = ConfigDict(frozen=True)

    min_caps_letters: Annotated[
        int, Field(description="Min letters before a line's case ratio is meaningful.")
    ] = 3
    caps_ratio: Annotated[
        float, Field(description="Share of letters that must be uppercase to read as all-caps.")
    ] = 0.8


class BodyProfileConfig(BaseModel):
    """Injectable thresholds measuring the body style a run is judged against."""

    model_config = ConfigDict(frozen=True)

    sample_pages: Annotated[
        int,
        Field(
            description="Divisor of the body-profile stride, not a page count: "
            "``step = page_count // this``, so a 190-page document is profiled on 64 pages "
            "and a 989-page one on 62, while anything shorter is profiled whole."
        ),
    ] = 60
    bold_ratio: Annotated[
        float,
        Field(description="Share of body characters that must be bold to call the body bold."),
    ] = 0.5
    default_size: Annotated[
        float, Field(description="Fallback baseline font size when the document has no text.")
    ] = 11.0


class CandidateConfig(BaseModel):
    """Injectable thresholds deciding whether a run is a heading candidate at all."""

    model_config = ConfigDict(frozen=True)

    min_chars: Annotated[
        int, Field(description="A heading candidate shorter than this holds no word; drop it.")
    ] = 2
    size_tolerance: Annotated[
        float, Field(description="Font size must exceed the baseline by this to read as larger.")
    ] = 0.5
    redundancy_drop_count: Annotated[
        int, Field(description="Drop a candidate whose text recurs this many times or more.")
    ] = 3
    soft_max_words: Annotated[
        int,
        Field(
            description="EXPERIMENTAL (may lower coverage): a run promoted only by a soft signal "
            "— a leading code or all-caps, not a bold/larger/underlined format deviation — is "
            "dropped as body text when it runs longer than this many words. Set too low, it "
            "drops real items whose descriptions run long, which the reconciler can never "
            "report again."
        ),
    ] = 40
    near_duplicate_prefix: Annotated[
        int,
        Field(
            description="EXPERIMENTAL (may lower coverage): leading characters keying the bucket "
            "near-duplicate candidates are collapsed within."
        ),
    ] = 40
    near_duplicate_ratio: Annotated[
        int,
        Field(
            description="EXPERIMENTAL (may lower coverage): two same-bucket candidates whose "
            "fuzz.ratio meets this are near duplicates, and the later one is dropped."
        ),
    ] = 95
    serial_min_items: Annotated[
        int,
        Field(
            ge=2,
            description="A numbered series promotes its items only when this many distinct "
            "items of one code family open a line (« BT-2026-0145 », « -0146 », « -0147 »).",
        ),
    ] = 3
    serial_step_share: Annotated[
        float,
        Field(
            gt=0,
            le=1,
            description="Share of a family's sorted item numbers that must follow the previous "
            "one by exactly 1. A list counts up; budget accounts (02-193-00-989), standards and "
            "dates recur at line openers without doing so, and stay out.",
        ),
    ] = 0.6
