"""Tests for ``Taxonomy``."""

from pathlib import Path

import pytest

from baseline_core.core.ml.data.taxonomy import LabelDefinition, Taxonomy


class TestTaxonomy:
    """Tests for loading, subsetting and the label mappings."""

    def test__init__when_names_repeated__should_raise_value_error(self) -> None:
        """Label names are unique."""
        label = LabelDefinition(name="bag", description="A bag.", status="retained")

        with pytest.raises(ValueError, match=r"duplicates: \['bag'\]"):
            Taxonomy(labels=[label, label])

    def test__load__should_read_the_yaml_labels(self, tmp_path: Path) -> None:
        """A YAML ``labels`` list becomes the taxonomy."""
        path = tmp_path / "taxonomy.yaml"
        path.write_text(
            "labels:\n  - name: bag\n    description: A bag.\n    status: retained\n",
            encoding="utf-8",
        )

        taxonomy = Taxonomy.load(path)

        assert taxonomy.descriptions == {"bag": "A bag."}

    def test__subset__should_keep_the_requested_order(self, taxonomy: Taxonomy) -> None:
        """The order of ``names`` defines the new ids."""
        subset = taxonomy.subset(["bag", "box"])

        assert subset.id2label == {0: "bag", 1: "box"}

    def test__subset__when_name_unknown__should_raise_value_error(
        self, taxonomy: Taxonomy
    ) -> None:
        """Names outside the taxonomy are an error."""
        with pytest.raises(ValueError, match=r"\['crate'\] are not in the taxonomy"):
            taxonomy.subset(["crate"])
