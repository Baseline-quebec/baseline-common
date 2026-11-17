import ast
import re
from functools import partial
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
def package() -> str:
    """The package these rules check, its name written here only for a copy in another repo.

    Returns:
        Its import name.
    """
    return "baseline_common"


@pytest.fixture
def source() -> str:
    """The `src` folder, the root the import graph is read from.

    Returns:
        Its absolute path, written as ArchUnitPython writes the paths of the files it reads.
    """
    return (Path(__file__).parents[1] / "src").absolute().as_posix()


@pytest.fixture
def root(source: str, package: str) -> str:
    """The start of every path pattern: the package's folder, so no parent folder's name matches.

    Args:
        source: The `src` folder.
        package: The package these rules check.

    Returns:
        A regular expression anchored on the package's folder, its trailing slash included.
    """
    return "^" + re.escape(f"{source}/{package}/")


class TestLayers:
    ORDER: ClassVar[list[list[str]]] = [
        ["framework"],
        ["models"],
        ["services"],
        ["orchestrators"],
        ["assembly"],
        ["api", "cli"],
    ]
    """The layers by rank, lowest first; a project adds its entry points beside `api` and `cli`."""
    LAYERS: ClassVar[list[str]] = [layer for rank in ORDER for layer in rank]
    """Every layer folder a file may live in."""

    def test__every_python_file__lives_in_a_layer(self, source: str, root: str) -> None:
        """Only the package's own `__init__.py` sits outside the layer folders the rules read."""
        layers = "|".join(self.LAYERS)
        rule = (
            project_files(source)
            .in_path(re.compile(root))
            .should()
            .be_in_path(re.compile(rf"{root}(__init__\.py$|({layers})/)"))
        )

        assert_passes(rule)

    def test__the_package_root__holds_no_folder_but_the_layers(
        self, source: str, package: str
    ) -> None:
        """Non-Python files such as `py.typed` may sit at the root; a folder there is a layer."""
        folders = {entry.name for entry in Path(source, package).iterdir() if entry.is_dir()}

        assert folders - {*self.LAYERS, "__pycache__"} == set()

    @pytest.mark.parametrize("layer", LAYERS)
    def test__a_layer__imports_no_layer_above_or_beside_it(
        self, source: str, root: str, layer: str
    ) -> None:
        """Dependencies point down only; the entry points, side by side, ignore each other."""
        rank = next(index for index, names in enumerate(self.ORDER) if layer in names)
        forbidden = [name for names in self.ORDER[rank:] for name in names if name != layer]
        rule = project_layers(source)

        for name in [layer, *forbidden]:
            rule = rule.layer(name).defined_by(re.compile(f"{root}{name}/"))

        assert_passes(rule.where_layer(layer).may_not_depend_on_layers(*forbidden))

    @staticmethod
    def imports_only_its_framework_module(file: FileInfo, source: str, package: str) -> bool:
        """Whether a framework file imports no framework module other than its own.

        Args:
            file: The file read by ArchUnitPython.
            source: The `src` folder.
            package: The package these rules check.

        Returns:
            True when every framework import it makes is of its own module.
        """
        own = Path(file.path).relative_to(Path(source, package, "framework")).parts[0]
        prefix = f"{package}.framework."
        names: list[str] = []

        for node in ast.walk(ast.parse(file.content)):
            if isinstance(node, ast.Import):
                names += [alias.name for alias in node.names]

            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                # `from <package>.framework import <module>` names the module in the alias only.
                names += [f"{node.module}.{alias.name}" for alias in node.names]

        imported = {
            name.removeprefix(prefix).split(".")[0] for name in names if name.startswith(prefix)
        }

        return imported <= {own}

    def test__each_framework_module__imports_no_other_one(
        self, source: str, package: str, root: str
    ) -> None:
        """A project copies one framework module alone, so it imports no other."""
        condition = partial(self.imports_only_its_framework_module, source=source, package=package)
        rule = (
            project_files(source)
            .in_path(re.compile(f"{root}framework/"))
            .should()
            .adhere_to(condition, "imports another framework module")
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
    SPECIALISED_PACKAGES: ClassVar[re.Pattern[str]] = re.compile(
        r"^baseline_(optimization|ml|llm)(\.|$)"
    )
    """The specialised repositories: they read baseline-common, never the reverse."""
    BASELINE_PACKAGES: ClassVar[list[str]] = [
        "baseline_common",
        "baseline_ml",
        "baseline_llm",
        "baseline_optimization",
    ]
    """Baseline's packages; a file names their modules, never one of them, nor `package`, alone."""

    @staticmethod
    def imports_no_package_whole(file: FileInfo, packages: set[str]) -> bool:
        """Whether a file imports modules by their path, never one of the packages alone.

        Args:
            file: The file read by ArchUnitPython.
            packages: The packages a file may not import whole.

        Returns:
            True when no `import <package>` nor `from <package> import <name>` names one of them.
        """
        modules: list[str | None] = []

        for node in ast.walk(ast.parse(file.content)):
            if isinstance(node, ast.Import):
                modules += [alias.name for alias in node.names]

            elif isinstance(node, ast.ImportFrom):
                modules += [node.module]

        return packages.isdisjoint(modules)

    def test__no_source_file__imports_a_package_whole(
        self, source: str, package: str, root: str
    ) -> None:
        """`from <package> import api` hides the layer it reaches from the layer rules."""
        packages = {package, *self.BASELINE_PACKAGES}
        condition = partial(self.imports_no_package_whole, packages=packages)
        rule = (
            project_files(source)
            .in_path(re.compile(root))
            .should()
            .adhere_to(condition, "imports a package whole instead of naming its module")
        )

        assert_passes(rule)

    def test__no_source_file__imports_a_client_project(self, source: str, root: str) -> None:
        """The framework and its examples are generic: no file imports a client project's package."""
        rule = (
            project_files(source)
            .in_path(re.compile(root))
            .should_not()
            .depend_on_external_modules()
            .matching(self.CLIENT_PACKAGES)
        )

        assert_passes(rule)

    def test__no_source_file__imports_a_specialised_repository(
        self, source: str, root: str
    ) -> None:
        """Dependencies between repositories point to baseline-common, never from it."""
        rule = (
            project_files(source)
            .in_path(re.compile(root))
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

    def test__a_package__re_exports_nothing(self, source: str, root: str) -> None:
        """An ``__init__.py`` holds a docstring only: importing a name loads only its own file."""
        rule = (
            project_files(source)
            .in_path(re.compile(root))
            .with_name("__init__.py")
            .should()
            .adhere_to(self.holds_only_a_docstring, "imports or defines something")
        )

        assert_passes(rule)

    def test__an_abstract_file__defines_the_abstract_class_it_is_named_after(
        self, source: str, root: str
    ) -> None:
        """A file `abstract_<name>.py` holds the ABC `Abstract…<Name>`."""
        rule = (
            project_files(source)
            .in_path(re.compile(root))
            .with_name(re.compile(r"^abstract_\w+\.py$"))
            .should()
            .adhere_to(self.defines_its_abstract_class, "defines no class Abstract…<Name>")
        )

        assert_passes(rule)
