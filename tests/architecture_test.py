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


# ---- Per repository: a copy of this file in another repository changes this block only. ----
class ThisRepository:
    """What sets one repository's copy apart; the rest of the file is the same in every copy."""

    PACKAGE: ClassVar[str] = "baseline_common"
    """The package these rules check."""
    ENTRY_POINTS: ClassVar[list[str]] = ["api", "cli"]
    """The top layers, side by side: none imports another; a project adds its own (`worker`)."""
    SPAWN_TARGETS: ClassVar[dict[str, list[str]]] = {}
    """Per entry point, the other entry points' functions it spawns (`worker.main.run`)."""
    SPECIALISED_PACKAGES: ClassVar[list[str]] = [
        "baseline_optimization",
        "baseline_ml",
        "baseline_llm",
    ]
    """The specialised repositories this one never imports, its own left out."""


# ---- End of the per-repository block. ----


@pytest.fixture
def package() -> str:
    """The package these rules check, named in `ThisRepository`.

    Returns:
        Its import name.
    """
    return ThisRepository.PACKAGE


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


class Imports:
    @staticmethod
    def names(code: str) -> list[str]:
        """The dotted names a file imports, `from <module> import <name>` as `<module>.<name>`.

        Args:
            code: The file's source code.

        Returns:
            Each imported name, relative imports left out.
        """
        names: list[str] = []

        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Import):
                names += [alias.name for alias in node.names]

            elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
                # `from <package>.framework import <module>` names the module in the alias only.
                names += [f"{node.module}.{alias.name}" for alias in node.names]

        return names

    @staticmethod
    def bindings(tree: ast.Module) -> dict[str, str]:
        """The full name each name a file imports stands for, under its alias if it has one.

        Args:
            tree: The file's syntax tree.

        Returns:
            Each local name with the dotted name it was imported as: `import dishka as di` binds
            `di` to `dishka`, `from dishka import Provider as Base` binds `Base`.
        """
        bound: dict[str, str] = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # `import a.b` binds `a` only; `import a.b as c` binds `c` to `a.b`.
                    local = alias.asname or alias.name.split(".")[0]
                    bound[local] = alias.name if alias.asname else local

            elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
                for alias in node.names:
                    bound[alias.asname or alias.name] = f"{node.module}.{alias.name}"

        return bound

    @staticmethod
    def resolve(node: ast.expr, bound: dict[str, str]) -> str | None:
        """The full name a name or an attribute chain stands for, through the file's imports.

        Args:
            node: The expression, `di.Provider` say.
            bound: The file's imported names, from `bindings`.

        Returns:
            Its dotted name (`dishka.Provider`), or None when it starts from no import.
        """
        if isinstance(node, ast.Name):
            return bound.get(node.id)

        if isinstance(node, ast.Attribute):
            owner = Imports.resolve(node.value, bound)
            return None if owner is None else f"{owner}.{node.attr}"

        return None

    @staticmethod
    def is_from_dishka(name: str | None, names: set[str]) -> bool:
        """Whether a full name is one of the names, taken from Dishka or one of its modules.

        Args:
            name: The full name, from `resolve`.
            names: The names Dishka exports that are looked for.

        Returns:
            True when it is.
        """
        return name is not None and name.split(".")[0] == "dishka" and name.split(".")[-1] in names

    @staticmethod
    def takes_from_dishka(file: FileInfo, names: set[str]) -> bool:
        """Whether a file imports or uses one of the names from Dishka, under any alias.

        Args:
            file: The file read by ArchUnitPython.
            names: The names Dishka exports that the file may not use.

        Returns:
            True when it does.
        """
        tree = ast.parse(file.content)
        bound = Imports.bindings(tree)
        used = [
            Imports.resolve(node, bound)
            for node in ast.walk(tree)
            if isinstance(node, ast.Name | ast.Attribute)
        ]

        return any(Imports.is_from_dishka(name, names) for name in [*bound.values(), *used])


class TestLayers:
    ORDER: ClassVar[list[str]] = ["framework", "models", "services", "orchestrators", "assembly"]
    """The layers below the entry points, lowest first."""
    LAYERS: ClassVar[list[str]] = [*ORDER, *ThisRepository.ENTRY_POINTS]
    """Every layer folder a file may live in."""
    DECLARING: ClassVar[set[str]] = {
        "Provider",
        "provide",
        "provide_all",
        "from_context",
        "alias",
        "decorate",
    }
    """What Dishka declares a provider or a factory with."""
    CONTAINERS: ClassVar[set[str]] = {"make_container", "make_async_container"}
    """What Dishka builds a container with."""

    def test__when_a_file_is_python__lives_in_a_layer(self, source: str, root: str) -> None:
        """Only the package's own `__init__.py` sits outside the layer folders the rules read."""
        layers = "|".join(self.LAYERS)
        rule = (
            project_files(source)
            .in_path(re.compile(root))
            .should()
            .be_in_path(re.compile(rf"{root}(__init__\.py$|({layers})/)"))
        )

        assert_passes(rule)

    def test__when_a_folder_is_at_the_package_root__is_a_layer(
        self, source: str, package: str
    ) -> None:
        """Non-Python files such as `py.typed` may sit at the root; a folder there is a layer."""
        folders = {entry.name for entry in Path(source, package).iterdir() if entry.is_dir()}

        assert folders - {*self.LAYERS, "__pycache__"} == set()

    @pytest.mark.parametrize("layer", ORDER)
    def test__when_a_layer_has_layers_above_it__never_imports_them(
        self, source: str, root: str, layer: str
    ) -> None:
        """Dependencies point down only, from the entry points to the framework."""
        above = [*self.ORDER[self.ORDER.index(layer) + 1 :], *ThisRepository.ENTRY_POINTS]
        rule = project_layers(source)

        for name in [layer, *above]:
            rule = rule.layer(name).defined_by(re.compile(f"{root}{name}/"))

        assert_passes(rule.where_layer(layer).may_not_depend_on_layers(*above))

    @staticmethod
    def names_only_its_spawn_targets(file: FileInfo, package: str, entry_point: str) -> bool:
        """Whether an entry point's file names no other entry point, but for its spawn targets.

        Args:
            file: The file read by ArchUnitPython.
            package: The package these rules check.
            entry_point: The entry point the file belongs to.

        Returns:
            True when each import of another entry point is a function `SPAWN_TARGETS` lists.
        """
        entry_points = ThisRepository.ENTRY_POINTS
        targets = ThisRepository.SPAWN_TARGETS.get(entry_point, [])
        others = [f"{package}.{other}" for other in entry_points if other != entry_point]
        allowed = {f"{package}.{target}" for target in targets}
        named = {
            name
            for name in Imports.names(file.content)
            if any(name == other or name.startswith(f"{other}.") for other in others)
        }

        return named <= allowed

    @pytest.mark.parametrize("entry_point", ThisRepository.ENTRY_POINTS)
    def test__when_an_entry_point_names_another__only_spawns_it(
        self, source: str, package: str, root: str, entry_point: str
    ) -> None:
        """Entry points ignore each other, but for the function one launches in a child process."""
        condition = partial(
            self.names_only_its_spawn_targets, package=package, entry_point=entry_point
        )
        rule = (
            project_files(source)
            .in_path(re.compile(f"{root}{entry_point}/"))
            .should()
            .adhere_to(condition, "imports another entry point beyond its spawn targets")
        )

        assert_passes(rule)

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
        imported = {
            name.removeprefix(prefix).split(".")[0]
            for name in Imports.names(file.content)
            if name.startswith(prefix)
        }

        return imported <= {own}

    def test__when_a_file_is_in_a_framework_module__imports_no_other_one(
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

    @classmethod
    def declares_a_provider(cls, file: FileInfo, package: str) -> bool:
        """Whether a file declares a Dishka provider, as a class or as an instance.

        Args:
            file: The file read by ArchUnitPython.
            package: The package these rules check.

        Returns:
            True when a class is based on one of Dishka's providers or on a provider of the
            assembly, or the file takes `Provider`, `provide` or another of `DECLARING` from
            Dishka. A class merely named `…Provider` (`WindmillSecretsProvider`) is none.
        """
        tree = ast.parse(file.content)
        bound = Imports.bindings(tree)
        bases = [
            Imports.resolve(base, bound) or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            for base in node.bases
        ]
        based = any(
            (base.startswith("dishka.") and base.endswith("Provider"))
            or base.startswith(f"{package}.assembly.")
            for base in bases
        )

        return based or Imports.takes_from_dishka(file, cls.DECLARING)

    @pytest.mark.parametrize("layer", [layer for layer in LAYERS if layer != "assembly"])
    def test__when_a_layer_is_not_the_assembly__declares_no_provider(
        self, source: str, package: str, root: str, layer: str
    ) -> None:
        """Providers live in `assembly/`; an entry point builds its container from them."""
        condition = partial(self.declares_a_provider, package=package)
        rule = (
            project_files(source)
            .in_path(re.compile(f"{root}{layer}/"))
            .should_not()
            .adhere_to(condition, "declares a Dishka provider")
        )

        assert_passes(rule)

    @pytest.mark.parametrize("layer", ORDER)
    def test__when_a_layer_is_below_the_entry_points__builds_no_container(
        self, source: str, root: str, layer: str
    ) -> None:
        """Only an entry point builds a container, in its `init_container()`."""
        condition = partial(Imports.takes_from_dishka, names=self.CONTAINERS)
        rule = (
            project_files(source)
            .in_path(re.compile(f"{root}{layer}/"))
            .should_not()
            .adhere_to(condition, "builds a Dishka container")
        )

        assert_passes(rule)

    @staticmethod
    def builds_its_container_without_argument(file: FileInfo) -> bool:
        """Whether every `init_container` a file defines takes no argument.

        Args:
            file: The file read by ArchUnitPython.

        Returns:
            True when none takes one: a test builds its own container instead.
        """
        functions = [
            node
            for node in ast.walk(ast.parse(file.content))
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == "init_container"
        ]

        return not any(
            function.args.posonlyargs
            or function.args.args
            or function.args.kwonlyargs
            or function.args.vararg
            or function.args.kwarg
            for function in functions
        )

    @classmethod
    def builds_containers_in_init_container(
        cls, node: ast.AST, function: ast.AST | None, bound: dict[str, str]
    ) -> bool:
        """Whether every container built under a node is built in a `def init_container()`.

        Args:
            node: The node read, its children included.
            function: The innermost function the node sits in, None at module level.
            bound: The file's imported names, from `Imports.bindings`.

        Returns:
            True when each use of `CONTAINERS`, under any alias, sits directly in a function
            named `init_container`: neither at module level nor in a `build(*overrides)`.
        """
        if isinstance(node, ast.Name | ast.Attribute) and Imports.is_from_dishka(
            Imports.resolve(node, bound), cls.CONTAINERS
        ):
            return (
                isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef)
                and function.name == "init_container"
            )

        functions = ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
        inner = node if isinstance(node, functions) else function

        return all(
            cls.builds_containers_in_init_container(child, inner, bound)
            for child in ast.iter_child_nodes(node)
        )

    @classmethod
    def builds_its_container_in_init_container(cls, file: FileInfo) -> bool:
        """Whether a file builds containers only in its `init_container()`.

        Args:
            file: The file read by ArchUnitPython.

        Returns:
            True when it does, or builds none.
        """
        tree = ast.parse(file.content)

        return cls.builds_containers_in_init_container(tree, None, Imports.bindings(tree))

    @pytest.mark.parametrize("entry_point", ThisRepository.ENTRY_POINTS)
    def test__when_an_entry_point_builds_a_container__builds_it_in_init_container_only(
        self, source: str, root: str, entry_point: str
    ) -> None:
        """One function builds the container, the one the app and a test both know."""
        rule = (
            project_files(source)
            .in_path(re.compile(f"{root}{entry_point}/"))
            .should()
            .adhere_to(
                self.builds_its_container_in_init_container,
                "builds a container outside init_container",
            )
        )

        assert_passes(rule)

    @pytest.mark.parametrize("entry_point", ThisRepository.ENTRY_POINTS)
    def test__when_an_entry_point_defines_init_container__it_takes_no_argument(
        self, source: str, root: str, entry_point: str
    ) -> None:
        """No parameter serves the tests only: a test builds its container and passes it."""
        rule = (
            project_files(source)
            .in_path(re.compile(f"{root}{entry_point}/"))
            .should()
            .adhere_to(
                self.builds_its_container_without_argument, "init_container takes an argument"
            )
        )

        assert_passes(rule)


class TestCycles:
    @staticmethod
    def marker(edge: ProjectedEdge) -> str:
        """The mark of an import made for typing only.

        Args:
            edge: The import, from one file to another.

        Returns:
            ` (TYPE_CHECKING)` when every import along it is a typing one, else nothing.
        """
        kinds = {kind for raw in edge.cumulated_edges for kind in raw.import_kinds}

        return " (TYPE_CHECKING)" if kinds == {ImportKind.TYPE_IMPORT} else ""

    @staticmethod
    def module_edges(source: str) -> set[tuple[str, str]]:
        """The imports between files that name a module, `from <package>.<sub> import <module>`.

        ArchUnitPython draws that import to `<sub>/__init__.py`: the module, and a cycle through
        it, would be lost.

        Args:
            source: The `src` folder.

        Returns:
            Each import as `(importer, imported)`, their paths written as ArchUnitPython does.
        """
        edges: set[tuple[str, str]] = set()

        for path in Path(source).rglob("*.py"):
            for name in Imports.names(path.read_text(encoding="utf-8")):
                module = Path(source, *name.split("."))

                for target in [module.with_name(f"{module.name}.py"), module / "__init__.py"]:
                    if target.is_file():
                        edges.add((path.as_posix(), target.as_posix()))

        return edges

    def test__when_imports_are_followed__the_package_has_no_cycle(self, source: str) -> None:
        """No file reaches itself again through its imports, `TYPE_CHECKING` ones included."""
        markers = {
            (edge.source_label, edge.target_label): self.marker(edge)
            for edge in project_edges(extract_graph(source), per_internal_edge())
        }
        edges = sorted({*markers, *self.module_edges(source)})
        labels = sorted({label for edge in edges for label in edge})
        numbers = {label: number for number, label in enumerate(labels)}

        # have_no_cycles recurses forever on some components (archunitpython 1.8.1); a strongly
        # connected component of more than one file is a cycle, so Tarjan alone decides.
        components = TarjanSCC().find_strongly_connected_components([
            NumberEdge(numbers[importer], numbers[imported]) for importer, imported in edges
        ])
        looping = {(labels[e.from_node], labels[e.to_node]) for scc in components for e in scc}
        cycles = [
            f"{Path(importer).relative_to(source)} -> {Path(imported).relative_to(source)}"
            f"{markers.get((importer, imported), '')}"
            for importer, imported in edges
            if (importer, imported) in looping
        ]

        assert cycles == [], "\n".join(cycles)


class TestForbiddenImports:
    CLIENT_PACKAGES: ClassVar[re.Pattern[str]] = re.compile(
        r"^(ecolopharm_sales|cegep_mv_schedule|project_assembly|recom_graphml"
        r"|benny_order_vision|munera_poc)(\.|$)"
    )
    """The client projects' packages, which no file imports."""
    HTTP_SERVERS: ClassVar[re.Pattern[str]] = re.compile(r"^(fastapi|starlette)(\.|$)")
    """The packages that serve HTTP: only the API imports them."""
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

    def test__when_a_file_is_in_the_package__imports_no_package_whole(
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

    def test__when_a_file_is_in_the_package__imports_no_client_project(
        self, source: str, root: str
    ) -> None:
        """The framework and its examples are generic: no file imports a client project's package."""
        rule = (
            project_files(source)
            .in_path(re.compile(root))
            .should_not()
            .depend_on_external_modules()
            .matching(self.CLIENT_PACKAGES)
        )

        assert_passes(rule)

    def test__when_a_file_is_in_the_package__imports_no_specialised_repository(
        self, source: str, root: str
    ) -> None:
        """Dependencies between repositories point to baseline-common, never from it."""
        specialised = "|".join(ThisRepository.SPECIALISED_PACKAGES)
        rule = (
            project_files(source)
            .in_path(re.compile(root))
            .should_not()
            .depend_on_external_modules()
            .matching(re.compile(rf"^({specialised})(\.|$)"))
        )

        assert_passes(rule)

    @classmethod
    def reads_baseline_through_framework_or_models(
        cls, file: FileInfo, source: str, package: str
    ) -> bool:
        """Whether a file reads other Baseline packages only through `framework/` or `models/`.

        Args:
            file: The file read by ArchUnitPython.
            source: The `src` folder.
            package: The package these rules check.

        Returns:
            True when it does; a framework file reads only their `framework/`.
        """
        layer = Path(file.path).relative_to(Path(source, package)).parts[0]
        readable = ["framework"] if layer == "framework" else ["framework", "models"]
        others = [other for other in cls.BASELINE_PACKAGES if other != package]

        # `from <other> import models` reads models/; importing a package whole has its own test.
        return all(
            any(f"{name}.".startswith(f"{other}.{part}.") for part in readable)
            for name in Imports.names(file.content)
            for other in others
            if name.startswith(f"{other}.")
        )

    def test__when_another_baseline_package_is_read__is_read_through_its_framework_or_models(
        self, source: str, package: str, root: str
    ) -> None:
        """A repository reads another's `framework/`, and its `models/` above the framework."""
        condition = partial(
            self.reads_baseline_through_framework_or_models, source=source, package=package
        )
        rule = (
            project_files(source)
            .in_path(re.compile(root))
            .should()
            .adhere_to(condition, "reads another Baseline package beyond framework/ or models/")
        )

        assert_passes(rule)

    def test__when_a_layer_is_not_the_api__knows_no_http(self, source: str, root: str) -> None:
        """Routes and requests stay in `api/`, and `framework/api/` that it imports: no other layer."""
        rule = (
            project_files(source)
            .in_path(re.compile(f"{root}(?!api/|framework/api/)"))
            .should_not()
            .depend_on_external_modules()
            .matching(self.HTTP_SERVERS)
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

    def test__when_a_package_has_an_init_file__re_exports_nothing(
        self, source: str, root: str
    ) -> None:
        """An ``__init__.py`` holds a docstring only: importing a name loads only its own file."""
        rule = (
            project_files(source)
            .in_path(re.compile(root))
            .with_name("__init__.py")
            .should()
            .adhere_to(self.holds_only_a_docstring, "imports or defines something")
        )

        assert_passes(rule)

    def test__when_a_file_is_named_abstract__defines_the_abstract_class_it_is_named_after(
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


class TestRepositories:
    READ: ClassVar[str] = "AbstractReadRepository"
    """The contract of a class that reads."""
    WRITE: ClassVar[str] = "AbstractWriteRepository"
    """The contract of a class that writes."""
    WHOLE: ClassVar[list[str]] = ["InMemoryRepository"]
    """Implements both contracts in one class: a test substitute, with no store to split."""

    @staticmethod
    def bases(folder: Path) -> dict[str, list[str]]:
        """The direct bases of each class defined under a folder, by name.

        Args:
            folder: The folder read, its subfolders included.

        Returns:
            Each class's name, with the names of its direct bases, their type arguments dropped.
        """
        classes: dict[str, list[str]] = {}

        for path in folder.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ClassDef):
                    bases = [b.value if isinstance(b, ast.Subscript) else b for b in node.bases]
                    classes[node.name] = [ast.unparse(base).rsplit(".", 1)[-1] for base in bases]

        return classes

    @staticmethod
    def ancestors(name: str, classes: dict[str, list[str]]) -> set[str]:
        """Every class a class inherits from, through the classes known by name.

        Args:
            name: The class.
            classes: The direct bases of each known class.

        Returns:
            Its ancestors' names.
        """
        found: set[str] = set()
        pending = list(classes.get(name, []))

        while pending:
            base = pending.pop()

            if base not in found:
                found.add(base)
                pending += classes.get(base, [])

        return found

    def test__when_a_store_has_repositories__provides_a_read_a_write_and_a_combined_class(
        self, source: str, package: str
    ) -> None:
        """Each store has a read class, a write class and a combined class inheriting both."""
        classes = self.bases(Path(source, package, "framework", "repository"))
        sides = {
            name: {self.READ, self.WRITE} & self.ancestors(name, classes)
            for name in classes
            if not name.startswith("Abstract")
        }
        readers = {name for name, side in sides.items() if side == {self.READ}}
        writers = {name for name, side in sides.items() if side == {self.WRITE}}
        combined = {name for name, side in sides.items() if len(side) == 2} - set(self.WHOLE)
        unpaired = [
            name
            for name in sorted(combined)
            if not (readers & set(classes[name]) and writers & set(classes[name]))
        ]
        halves = {base for name in combined for base in classes[name]}

        assert unpaired == [], "combined without both a read class and a write class"
        assert sorted((readers | writers) - halves) == [], (
            "read or write class with no combined one"
        )
