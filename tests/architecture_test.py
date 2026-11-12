import ast
import re
from pathlib import Path
from typing import ClassVar

import pytest
from archunitpython import assert_passes, extract_graph, project_files, project_layers
from archunitpython.common.extraction.graph import ImportKind
from archunitpython.common.projection.cycles.model import NumberEdge
from archunitpython.common.projection.cycles.tarjan_scc import TarjanSCC
from archunitpython.common.projection.edge_projections import per_internal_edge
from archunitpython.common.projection.project_edges import project_edges
from archunitpython.common.projection.types import ProjectedEdge
from archunitpython.files.assertion.custom_file_logic import FileInfo


@pytest.fixture
def source() -> str:
    """The `src` folder, the root the import graph is read from.

    Returns:
        Its path.
    """
    return str(Path(__file__).parents[1] / "src")


class TestLayers:
    CORE: ClassVar[str] = "*/baseline_common/core/*"
    ORDER: ClassVar[list[str]] = ["core", "data_schema", "service", "orchestrator", "api"]
    CORE_PACKAGE: ClassVar[str] = "baseline_common.core."

    @pytest.mark.parametrize("layer", ["core", "data_schema", "service", "orchestrator"])
    def test__a_layer__never_imports_a_layer_above_it(self, source: str, layer: str) -> None:
        """Dependencies point down only: api → orchestrator → service → data_schema → core."""
        above = self.ORDER[self.ORDER.index(layer) + 1 :]
        rule = project_layers(source)

        for name in [layer, *above]:
            rule = rule.layer(name).defined_by(f"*/baseline_common/{name}/*")

        assert_passes(rule.where_layer(layer).may_not_depend_on_layers(*above))

    @classmethod
    def imports_only_its_core_module(cls, file: FileInfo) -> bool:
        """Whether a core file imports no core module other than its own.

        Args:
            file: The file read by ArchUnitPython.

        Returns:
            True when every core import it makes is of its own module.
        """
        inside = Path(file.path).as_posix().split("/baseline_common/core/", 1)[1]
        own = inside.split("/", 1)[0]
        names: list[str] = []

        for node in ast.walk(ast.parse(file.content)):
            if isinstance(node, ast.Import):
                names += [alias.name for alias in node.names]

            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names += [node.module]

        imported = {name.split(".")[2] for name in names if name.startswith(cls.CORE_PACKAGE)}

        return imported <= {own}

    def test__each_core_module__imports_no_other_core_module(self, source: str) -> None:
        """A project copies one core module alone, so it imports no other."""
        rule = (
            project_files(source)
            .in_path(self.CORE)
            .should()
            .adhere_to(self.imports_only_its_core_module, "imports another core module")
        )

        assert_passes(rule)


class TestCycles:
    @staticmethod
    def describe(edge: ProjectedEdge, source: str) -> str:
        """One import of a cycle, its typing-only imports marked.

        Args:
            edge: The import, from one file of the cycle to another.
            source: The `src` folder, cut from the paths.

        Returns:
            The import as `importer -> imported`.
        """
        kinds = {kind for raw in edge.cumulated_edges for kind in raw.import_kinds}
        marker = " (TYPE_CHECKING)" if kinds == {ImportKind.TYPE_IMPORT} else ""
        importer = Path(edge.source_label).relative_to(source)
        imported = Path(edge.target_label).relative_to(source)

        return f"{importer} -> {imported}{marker}"

    def test__the_package__has_no_import_cycle(self, source: str) -> None:
        """No file reaches itself again through its imports, `TYPE_CHECKING` ones included."""
        edges = project_edges(extract_graph(source), per_internal_edge())
        labels = sorted(
            {edge.source_label for edge in edges} | {edge.target_label for edge in edges}
        )
        numbers = {label: number for number, label in enumerate(labels)}

        # have_no_cycles recurses forever on some components (archunitpython 1.8.1); a strongly
        # connected component of more than one file is a cycle, so Tarjan alone decides.
        components = TarjanSCC().find_strongly_connected_components([
            NumberEdge(numbers[edge.source_label], numbers[edge.target_label]) for edge in edges
        ])
        looping = {(labels[e.from_node], labels[e.to_node]) for scc in components for e in scc}
        cycles = sorted(
            self.describe(edge, source)
            for edge in edges
            if (edge.source_label, edge.target_label) in looping
        )

        assert cycles == [], "\n".join(cycles)


class TestForbiddenImports:
    CLIENT_PACKAGES: ClassVar[re.Pattern[str]] = re.compile(
        r"^(ecolopharm_sales|cegep_mv_schedule|project_assembly|recom_graphml"
        r"|benny_order_vision|munera_poc)(\.|$)"
    )
    # the specialised repositories read baseline-common, never the reverse
    SPECIALISED_PACKAGES: ClassVar[re.Pattern[str]] = re.compile(
        r"^baseline_(optimization|ml|llm)(\.|$)"
    )

    def test__no_source_file__imports_a_client_project(self, source: str) -> None:
        """Core and its examples are generic: no file imports a client project's package."""
        rule = (
            project_files(source)
            .in_path("*/baseline_common/*")
            .should_not()
            .depend_on_external_modules()
            .matching(self.CLIENT_PACKAGES)
        )

        assert_passes(rule)

    def test__no_source_file__imports_a_specialised_repository(self, source: str) -> None:
        """Dependencies between repositories point to baseline-common, never from it."""
        rule = (
            project_files(source)
            .in_path("*/baseline_common/*")
            .should_not()
            .depend_on_external_modules()
            .matching(self.SPECIALISED_PACKAGES)
        )

        assert_passes(rule)


class TestNaming:
    @staticmethod
    def defines_its_abstract_class(file: FileInfo) -> bool:
        """Whether `abstract_<name>.py` defines a class `Abstract…<Name>`, as in the README.

        Args:
            file: The file read by ArchUnitPython.

        Returns:
            True when it does.
        """
        suffix = "".join(part.capitalize() for part in file.name.split("_")[1:])
        tree = ast.parse(file.content)

        return any(
            isinstance(node, ast.ClassDef)
            and node.name.startswith("Abstract")
            and node.name.endswith(suffix)
            for node in tree.body
        )

    @staticmethod
    def holds_only_a_docstring(file: FileInfo) -> bool:
        """Whether an ``__init__.py`` holds nothing but its docstring.

        Args:
            file: The file read by ArchUnitPython.

        Returns:
            True when it does.
        """
        body = ast.parse(file.content).body

        return all(
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) for node in body
        )

    def test__a_package__re_exports_nothing(self, source: str) -> None:
        """An ``__init__.py`` holds a docstring only: importing a name loads only its own file."""
        rule = (
            project_files(source)
            .in_path("*/baseline_common/*")
            .with_name("__init__.py")
            .should()
            .adhere_to(self.holds_only_a_docstring, "imports or defines something")
        )

        assert_passes(rule)

    def test__an_abstract_file__defines_the_abstract_class_it_is_named_after(
        self, source: str
    ) -> None:
        """A file `abstract_<name>.py` holds the ABC `Abstract…<Name>`."""
        rule = (
            project_files(source)
            .in_path("*/baseline_common/*")
            .with_name(re.compile(r"^abstract_\w+\.py$"))
            .should()
            .adhere_to(self.defines_its_abstract_class, "defines no class Abstract…<Name>")
        )

        assert_passes(rule)
