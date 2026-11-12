"""Writes a model's kernel: ``kernel.py`` then a subclass generated from the model's shapes."""

import ast
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from baseline_core.core.optimization.problem_formulation.declaration import (
    Aggregate,
    Member,
    Position,
    Size,
    TargetRead,
    ValueRead,
)
from baseline_core.core.optimization.problem_formulation.expression import (
    Arithmetic,
    Comparison,
    Constant,
    Expr,
    IfThenElse,
    Logical,
    Lookup,
    Not,
)
from baseline_core.core.optimization.solver.binding.bound_collection import (
    BoundFilteredSet,
    ComputedSubset,
    FixedSubset,
    PartitionPart,
)
from baseline_core.core.optimization.solver.kernel import kernel, model_kernel
from baseline_core.core.optimization.solver.kernel.shape import Template, TermKind


@dataclass(frozen=True, slots=True)
class GeneratedKernel:
    """One model's kernel module, and the arrays its generated subclass reads."""

    source: str
    """The hand-written kernel, then the generated subclass."""
    arrays: dict[str, NDArray[np.int64]]
    """Every array the generated subclass reads, by name."""


class KernelSource:
    """One model's kernel source and every array it reads, written from its terms' shapes."""

    BIG: ClassVar[int] = 1 << 40
    """Stands for infinity in the integer arrays, and for "no member" in a minimum or maximum."""

    def __init__(self, templates: list[Template], n_elements: int, n_sets: int) -> None:
        """Take the inputs of one module; ``write`` fills the working state below.

        Args:
            templates: One per shape of term.
            n_elements: Number of elements.
            n_sets: Number of sets.
        """
        self.templates = templates
        self.n_elements = n_elements
        self.n_sets = n_sets

        self.template: Template
        """The template whose code is being written, set by ``write`` before any is written."""

        self.registry: list[NDArray[Any]] = []
        self.arrays: dict[str, NDArray[np.int64]] = {}
        self.n_temps = 0
        self.loop_count = 0
        self.fixed_ids: dict[int, int] = {}
        self.fixed_flat: list[int] = []
        self.fixed_offsets: list[int] = [0]
        self.next_const = 0
        self.next_collection = 0
        self.preamble: list[str] = []
        self.env: dict[int, str] = {}
        self.sequence_env: dict[int, str] = {}

    @classmethod
    def write(
        cls, templates: list[Template], n_elements: int, n_sets: int, ranks: NDArray[np.int64]
    ) -> GeneratedKernel:
        """Write the module of one model: the hand-written kernel, then its generated subclass.

        Args:
            templates: One per shape of term.
            n_elements: Number of elements.
            n_sets: Number of sets.
            ranks: Per element, its rank in its key's order: how a sorted list breaks ties.

        Returns:
            The source, and every array the generated subclass reads.
        """
        writer = cls(templates, n_elements, n_sets)

        for template in templates:
            for subset in template.subsets:
                if id(subset) not in writer.fixed_ids:
                    writer.fixed_ids[id(subset)] = len(writer.fixed_offsets) - 1
                    writer.fixed_flat.extend(subset.elements)
                    writer.fixed_offsets.append(len(writer.fixed_flat))

        writer.arrays["rank"] = ranks
        writer.arrays["CS_flat"] = np.array(writer.fixed_flat or [0], dtype=np.int64)
        writer.arrays["CS_off"] = np.array(writer.fixed_offsets, dtype=np.int64)

        blocks: dict[str, list[str]] = {
            "around": [],
            "full": [],
            "around_set": [],
            "constants": [],
            "violations": [],
        }

        for template in templates:
            writer.template = template
            writer._prepare()

            if not template.shape.reads_decision:
                blocks["constants"] += writer._full()
                continue

            blocks["around"] += writer._around()
            blocks["full"] += writer._guard(writer._full())
            blocks["around_set"] += writer._around_set()

            if template.violation:
                blocks["violations"] += writer._full(recording=True)

        for number, data in enumerate(writer.registry):
            writer.arrays[f"D{number}"] = writer._integers(data, "model data")

        source = Path(kernel.__file__).with_suffix(".py").read_text() + writer._subclass(blocks)
        return GeneratedKernel(source, writer.arrays)

    def _subclass(self, blocks: dict[str, list[str]]) -> str:
        """The generated subclass: its arrays, its constructor and its five blocks.

        Args:
            blocks: The lines of each block.

        Returns:
            The source.
        """
        template = Path(model_kernel.__file__).with_suffix(".py").read_text()
        bound = {
            node.name.removeprefix("_"): self._method_locals(node)
            for node in ast.walk(ast.parse(template))
            if isinstance(node, ast.FunctionDef)
        }

        slots = {
            "fields": [f"    {name}: cython.long[::1]" for name in self.arrays],
            "binds": [f'        self.{name} = arrays["{name}"]' for name in self.arrays],
            **{
                name: self._declarations(lines, bound[name]) + lines
                for name, lines in blocks.items()
            },
        }

        lines = template.splitlines()
        # the template's imports are for the IDE: the generated class follows kernel.py's own
        out: list[str] = []

        for line in lines[lines.index("@cython.cclass") :]:
            slot = line.strip().removeprefix("# generated ")
            out += slots[slot] if slot != line.strip() else [line]

        return "\n\n" + "\n".join(out) + "\n"

    @staticmethod
    def _method_locals(method: ast.FunctionDef) -> set[str]:
        """The names a template method binds itself: its arguments and its annotated locals.

        Args:
            method: The method.

        Returns:
            The names.
        """
        arguments = {argument.arg for argument in method.args.args}
        return arguments | {
            node.target.id
            for node in method.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }

    @staticmethod
    def _declarations(lines: list[str], bound: set[str]) -> list[str]:
        """Declare as C integers the locals a block assigns that its method does not bind.

        Args:
            lines: The block.
            bound: The names its method binds itself.

        Returns:
            The declarations.
        """
        tree = ast.parse(textwrap.dedent("\n".join(lines)))
        assigned = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        return [f"        {name}: cython.long" for name in sorted(assigned - bound)]

    def _prepare(self) -> None:
        """Fill the current template's arrays: constants, collections, instances per element."""
        template = self.template
        prefix = f"T{template.index}"

        self.arrays[f"{prefix}_P"] = (
            self._integers(template.constants, f"the numbers of {template.name!r}")
            if template.constants.size
            else np.zeros(1, dtype=np.int64)
        )

        ids = -1 - template.cells
        fixed = template.cells >= 0
        ranks = np.array(
            [self.fixed_ids[id(subset)] for subset in template.subsets], dtype=np.int64
        )
        ids[fixed] = ranks[template.cells[fixed]]
        self.arrays[f"{prefix}_C"] = ids.ravel() if ids.size else np.zeros(1, dtype=np.int64)

        count = template.n_instances
        reads = template.reads
        readers = np.repeat(np.arange(count), np.diff(template.read_offsets))

        if template.has_free_members:
            members = template.members
            owners = np.repeat(np.arange(count), np.diff(template.member_offsets))

            if template.shape.on_sets:
                self._inverse(f"{prefix}_", -1 - members, owners, self.n_sets)
            else:
                # an element its instance also reads is counted with the whole instance instead
                width = self.n_elements + 1
                fixed = reads >= 0
                kept = ~np.isin(owners * width + members, readers[fixed] * width + reads[fixed])
                self._inverse(f"{prefix}_", members[kept], owners[kept], self.n_elements)

        if template.reads_elements:
            fixed = reads >= 0
            self._inverse(f"{prefix}_r", reads[fixed], readers[fixed], self.n_elements)

        if template.reads_sets:
            decided = reads < 0
            self._inverse(f"{prefix}_s", -1 - reads[decided], readers[decided], self.n_sets)

        self.arrays[f"{prefix}_stamp"] = np.full(max(count, 1), -1, dtype=np.int64)
        self.arrays[f"{prefix}_n"] = np.array([count], dtype=np.int64)

    @classmethod
    def _integers(cls, data: object, what: str) -> NDArray[np.int64]:
        """Numbers as the kernel's integers, an infinity as ``BIG``, flattened.

        Args:
            data: The numbers.
            what: What they are, for the error.

        Returns:
            The integers.

        Raises:
            ValueError: A finite number is not integer: the kernel would truncate it.
        """
        values = np.asarray(data, dtype=np.float64)
        finite = np.isfinite(values)
        fractional = values[finite & (values != np.round(values))]

        if fractional.size:
            msg = f"{what} must be integers, the kernel counts in integers: got {fractional[0]}"
            raise ValueError(msg)

        clipped = np.where(finite, values, np.sign(values) * cls.BIG)
        return np.ascontiguousarray(clipped, dtype=np.int64).ravel()

    def _inverse(
        self, name: str, keys: NDArray[np.int64], instances: NDArray[np.int64], size: int
    ) -> None:
        """Register the instances of each key (element or set), sorted, as ``{name}inv``/``off``.

        Args:
            name: The arrays' prefix.
            keys: The key of each pair.
            instances: The instance of each pair.
            size: How many keys there are.
        """
        stride = max(self.template.n_instances, 1)
        pairs = np.unique(keys * stride + instances)
        self.arrays[f"{name}inv"] = pairs % stride if pairs.size else np.zeros(1, dtype=np.int64)
        self.arrays[f"{name}off"] = np.concatenate([
            [0],
            np.cumsum(np.bincount(pairs // stride, minlength=size)),
        ]).astype(np.int64)

    def _accumulate(self, members: list[str], *, recording: bool) -> list[str]:
        """The statements adding one evaluation to the count, or recording it when it is violated.

        Args:
            members: The members of the evaluation, as generated names.
            recording: Whether the violated evaluations are recorded instead of counted.

        Returns:
            The statements.
        """
        if recording:
            template = self.template
            record = f"[{template.index}, inst, [{', '.join(members)}], {template.name_index}]"
            return ["if expr_v == 0:", f"    found.append({record})"]

        counted = "(1 if expr_v == 0 else 0)" if self.template.violation else "expr_v"
        return [f"self.cnt[{self.template.name_index}] += {counted}"]

    def _body(
        self,
        env: dict[int, str],
        members: list[str],
        sequence_env: dict[int, str] | None = None,
        *,
        recording: bool = False,
    ) -> list[str]:
        """The statements evaluating one instance and adding it to the count, unindented.

        Args:
            env: The generated name of each free member.
            members: The members of the evaluation, as generated names.
            sequence_env: The generated name of each position of a sorted list.
            recording: Whether the violated evaluations are recorded instead of counted.

        Returns:
            The lines.
        """
        self.next_const = 0
        self.next_collection = 1 if self.template.shape.kind != TermKind.GROUP else 0
        self.preamble = []
        self.env, self.sequence_env = dict(env), sequence_env or {}

        expr = self._gen(self.template.expr)
        return [
            *self.preamble,
            f"expr_v = {expr}",
            *self._accumulate(members, recording=recording),
        ]

    def _mask(self) -> str | None:
        """The generated name of the domain's mask, if the domain is filtered.

        Returns:
            The array name, or ``None``.
        """
        mask = self.template.shape.mask
        return f"self.D{self._data(mask)}" if mask is not None else None

    def _guard(self, lines: list[str]) -> list[str]:
        """The lines, run only when the template's constraint is active.

        Args:
            lines: The block.

        Returns:
            The guarded block (no lines for an empty block).
        """
        if not lines:
            return []

        return [f"        if self.act[{self.template.name_index}] != 0:"] + [
            "    " + line for line in lines
        ]

    def _around(self) -> list[str]:
        """The block counting the evaluations an element (``j``) reads, or its sets hold.

        Returns:
            The lines.
        """
        template = self.template
        out = ["        self.sid += 1"] if self._stamped else []

        if template.reads_elements:
            out += self._readers(
                f"self.T{template.index}_r", "j", "        ", stamped=self._stamped
            )

        if template.reads_sets:
            out += [
                "        r0_ = self.target[j]",
                "        nr_ = 1",
                "        if r_other >= 0 and r_other != r0_:",
                "            r1_ = r_other",
                "            nr_ = 2",
                *self._only_masked("        "),
                "        for g_ in range(nr_):",
                "            r_ = r0_ if g_ == 0 else r1_",
                *self._readers(
                    f"self.T{template.index}_s", "r_", "            ", stamped=self._stamped
                ),
            ]

        if template.has_free_members:
            out += self._free_around()

        return self._guard(out)

    @property
    def _stamped(self) -> bool:
        """Whether an instance can be reached twice around one element, so is stamped."""
        template = self.template
        # a free member on a decided set cannot leave out, in advance, the elements it also reads
        on_set_reads = (
            template.has_free_members and template.shape.on_sets and template.reads_elements
        )
        return on_set_reads or (
            template.reads_sets > 0
            and (template.has_free_members or template.reads_elements or template.reads_sets > 1)
        )

    def _only_masked(self, pad: str) -> list[str]:
        """The statement skipping the sets of an element no collection of the term holds.

        Args:
            pad: The indentation.

        Returns:
            The lines (none unless all the term's collections share one mask).

        Raises:
            ValueError: A term filtered by one mask that has no mask.
        """
        if not self.template.shape.only_masked:
            return []

        only_mask = self.template.shape.masks[0]

        if only_mask is None:
            msg = "a term filtered by one mask has that mask"
            raise ValueError(msg)

        return [f"{pad}if self.D{self._data(only_mask)}[j] == 0:", f"{pad}    nr_ = 0"]

    def _readers(self, name: str, key: str, pad: str, *, stamped: bool) -> list[str]:
        """The loop evaluating, whole, every instance listed under ``key`` in ``{name}inv``.

        Args:
            name: The inverse's prefix.
            key: The generated element or set.
            pad: The indentation.
            stamped: Whether an instance already evaluated in this pass is skipped.

        Returns:
            The lines.
        """
        out = [
            f"{pad}for q_ in range({name}off[{key}], {name}off[{key} + 1]):",
            f"{pad}    inst = {name}inv[q_]",
        ]

        if stamped:
            stamp = f"self.T{self.template.index}_stamp[inst]"
            out += [
                f"{pad}    if {stamp} == self.sid:",
                f"{pad}        continue",
                f"{pad}    {stamp} = self.sid",
            ]

        return out + self._whole(pad + "    ")

    def _free_around(self) -> list[str]:
        """The loop evaluating the free evaluations of ``j`` (a member or pair term).

        Returns:
            The lines.
        """
        template = self.template
        inverse, offsets = f"self.T{template.index}_inv", f"self.T{template.index}_off"
        mask = self._mask()
        domain = f"self.T{template.index}_C[inst * {len(template.shape.collections)}]"
        out: list[str] = []

        if template.shape.on_sets:
            out += ["        r_ = self.target[j]"]

            if mask:
                out += [f"        if {mask}[j] == 0:", "            r_ = -1"]

            out += [
                "        if r_ >= 0:",
                f"            for q_ in range({offsets}[r_], {offsets}[r_ + 1]):",
                f"                inst = {inverse}[q_]",
            ]
            pad = "                "
        else:
            out += [
                f"        for q_ in range({offsets}[j], {offsets}[j + 1]):",
                f"            inst = {inverse}[q_]",
            ]
            pad = "            "

        if self._stamped:
            out += [
                f"{pad}if self.T{template.index}_stamp[inst] == self.sid:",
                f"{pad}    continue",
            ]

        return out + self._per_instance(pad, domain, mask, each_once=False)

    def _around_set(self) -> list[str]:
        """The block counting the evaluations a set of elements reads, each once.

        Returns:
            The lines (none for whole decided sets: the kernel answers with the full count).
        """
        template = self.template

        if template.set_level:
            return []

        loop = ["        for jj_ in range(n_elements):", "            j = self.elements[jj_]"]
        out: list[str] = []

        if template.reads_elements or template.reads_sets:
            out += loop

            if template.reads_elements:
                out += self._readers(
                    f"self.T{template.index}_r", "j", "            ", stamped=True
                )

            if template.reads_sets:
                out += [
                    "            r0_ = self.target[j]",
                    "            nr_ = 1",
                    "            if self.others[jj_] >= 0 and self.others[jj_] != r0_:",
                    "                r1_ = self.others[jj_]",
                    "                nr_ = 2",
                    *self._only_masked("            "),
                    "            for g_ in range(nr_):",
                    "                r_ = r0_ if g_ == 0 else r1_",
                    *self._readers(
                        f"self.T{template.index}_s", "r_", "                ", stamped=True
                    ),
                ]

        if template.has_free_members:
            inverse, offsets = f"self.T{template.index}_inv", f"self.T{template.index}_off"
            mask_name = self._mask()
            domain = f"self.T{template.index}_C[inst * {len(template.shape.collections)}]"
            key = "self.target[j]" if template.shape.on_sets else "j"

            out += [*loop, "            if self.absent[j] != 0:", "                continue"]

            if template.shape.on_sets and mask_name:
                out += [f"            if {mask_name}[j] == 0:", "                continue"]

            out += [
                f"            for q_ in range({offsets}[{key}], {offsets}[{key} + 1]):",
                f"                inst = {inverse}[q_]",
            ]

            if template.reads_elements or template.reads_sets:
                out += [
                    f"                if self.T{template.index}_stamp[inst] == self.sid:",
                    "                    continue",
                ]

            out += self._per_instance("                ", domain, mask_name, each_once=True)

        return self._guard(out)

    def _full(self, *, recording: bool = False) -> list[str]:
        """The block evaluating every instance.

        Args:
            recording: Whether the violated evaluations are recorded instead of counted.

        Returns:
            The lines.
        """
        out = [f"        for inst in range(self.T{self.template.index}_n[0]):"]
        return out + self._whole("            ", recording=recording)

    def _whole(self, pad: str, *, recording: bool = False) -> list[str]:
        """The statements evaluating every evaluation of one instance (``inst``).

        Args:
            pad: The indentation.
            recording: Whether the violated evaluations are recorded instead of counted.

        Returns:
            The lines.
        """
        mask = self._mask()
        domain = f"self.T{self.template.index}_C[inst * {len(self.template.shape.collections)}]"
        free = [member.symbol for member in self.template.shape.free]

        if self.template.shape.kind == TermKind.MEMBER:
            body = self._body({free[0]: "a_"}, ["a_"], recording=recording)
            return [
                pad + line
                for line in self._iterate(self.template.shape.domain, domain, "a_", body)
            ]

        if self.template.shape.kind == TermKind.PAIR:
            return self._full_pairs(domain, mask, pad, recording=recording)

        if self.template.shape.kind == TermKind.GROUP:
            return [pad + line for line in self._body({}, [], recording=recording)]

        return self._sequence(pad, domain, recording=recording)

    def _full_pairs(
        self, domain: str, mask: str | None, pad: str, *, recording: bool
    ) -> list[str]:
        """The loop over every pair of a pair template's instance.

        Args:
            domain: The generated reference to the instance's collection.
            mask: The domain's mask, if any.
            pad: The indentation.
            recording: Whether the violated evaluations are recorded instead of counted.

        Returns:
            The lines.
        """
        free = [member.symbol for member in self.template.shape.free]
        inner = [
            "x_ = a_ if a_ < b_ else b_",
            "y_ = b_ if a_ < b_ else a_",
            *self._body({free[0]: "x_", free[1]: "y_"}, ["x_", "y_"], recording=recording),
        ]
        base = self.template.shape.base

        if isinstance(base, FixedSubset):
            out = [
                f"{pad}for pa_ in range(self.CS_off[{domain}], self.CS_off[{domain} + 1]):",
                f"{pad}    a_ = self.CS_flat[pa_]",
            ]

            if base.decision:
                out += [f"{pad}    if self.absent[a_] != 0:", f"{pad}        continue"]

            if mask:
                out += [f"{pad}    if {mask}[a_] == 0:", f"{pad}        continue"]

            out += [
                f"{pad}    for pb_ in range(pa_ + 1, self.CS_off[{domain} + 1]):",
                f"{pad}        b_ = self.CS_flat[pb_]",
            ]

            if base.decision:
                out += [f"{pad}        if self.absent[b_] != 0:", f"{pad}            continue"]

            if mask:
                out += [f"{pad}        if {mask}[b_] == 0:", f"{pad}            continue"]

            return out + [f"{pad}        {line}" for line in inner]

        first_kept = f"{mask}[a_] != 0" if mask else "True"
        second_kept = f"{mask}[b_] != 0" if mask else "True"

        return [
            f"{pad}a_ = self.head[{domain}]",
            f"{pad}while a_ >= 0:",
            f"{pad}    b_ = self.nxt[a_]",
            f"{pad}    if {first_kept}:",
            f"{pad}        while b_ >= 0:",
            f"{pad}            if {second_kept}:",
            *[f"{pad}                {line}" for line in inner],
            f"{pad}            b_ = self.nxt[b_]",
            f"{pad}    a_ = self.nxt[a_]",
        ]

    def _per_instance(
        self, pad: str, domain: str, mask: str | None, *, each_once: bool
    ) -> list[str]:
        """The statements evaluating one instance an element (``j``) can see.

        Args:
            pad: The indentation.
            domain: The generated reference to the instance's collection.
            mask: The domain's mask, if any.
            each_once: Whether a pair of two elements of the set is counted once, not twice.

        Returns:
            The lines.
        """
        free = [member.symbol for member in self.template.shape.free]

        if self.template.shape.kind == TermKind.MEMBER:
            return [pad + line for line in self._body({free[0]: "j"}, ["j"])]

        if self.template.shape.kind == TermKind.GROUP:
            return [pad + line for line in self._body({}, [])]

        if self.template.shape.kind == TermKind.SEQUENCE:
            return self._sequence(pad, domain)

        skip = "k_ == j" + (" or (self.scope[k_] != 0 and k_ < j)" if each_once else "")

        if mask:
            skip += f" or {mask}[k_] == 0"

        body = [
            f"if {skip}:",
            "    pass",
            "else:",
            "    x_ = j if j < k_ else k_",
            "    y_ = k_ if j < k_ else j",
            *["    " + line for line in self._body({free[0]: "x_", free[1]: "y_"}, ["x_", "y_"])],
        ]

        return [pad + line for line in self._iterate(self.template.shape.base, domain, "k_", body)]

    def _sequence(self, pad: str, domain: str, *, recording: bool = False) -> list[str]:
        """Insertion-sort the instance's members by value, then evaluate each consecutive pair.

        Args:
            pad: The indentation.
            domain: The generated reference to the instance's collection.
            recording: Whether the violated evaluations are recorded instead of counted.

        Returns:
            The lines.

        Raises:
            ValueError: The template has no sorted list.
        """
        sorted_list = self.template.shape.sorted_list

        if sorted_list is None:
            msg = "a sequence term reads a sorted list"
            raise ValueError(msg)

        previous = "self.buf[p_ - 1]"
        gather = self._iterate(sorted_list.base, domain, "e_", ["self.buf[m_] = e_", "m_ += 1"])

        out = [
            f"{pad}m_ = 0",
            *[pad + line for line in gather],
            f"{pad}for i_ in range(1, m_):",
            f"{pad}    e_ = self.buf[i_]",
            f"{pad}    kx_ = self.start[e_]",
            f"{pad}    p_ = i_",
            (
                f"{pad}    while p_ > 0 and (self.start[{previous}] > kx_"
                f" or (self.start[{previous}] == kx_ and self.rank[{previous}] > self.rank[e_])):"
            ),
            f"{pad}        self.buf[p_] = {previous}",
            f"{pad}        p_ -= 1",
            f"{pad}    self.buf[p_] = e_",
            f"{pad}for i_ in range(1, m_):",
        ]

        consecutive = {0: "self.buf[i_]", -1: "self.buf[i_ - 1]"}
        members = [consecutive[-1], consecutive[0]]
        body = self._body({}, members, consecutive, recording=recording)
        return out + [pad + "    " + line for line in body]

    def _iterate(
        self, collection: object, collection_id: str, var: str, body: list[str]
    ) -> list[str]:
        """Loop ``var`` over a fixed subset, decided set or family read at an index, maybe masked.

        Args:
            collection: The collection.
            collection_id: The generated reference to its id.
            var: The generated name of the loop variable.
            body: The loop body.

        Returns:
            The lines, unindented.
        """
        mask = None

        if isinstance(collection, BoundFilteredSet):
            mask = f"self.D{self._data(collection.mask)}"
            collection = collection.base

        inner = ([f"if {mask}[{var}] != 0:"] + ["    " + line for line in body]) if mask else body

        if isinstance(collection, ComputedSubset):
            iterator = self._iterator()
            flat, offsets = self._data(collection.flat), self._data(collection.offsets)
            return [
                (
                    f"for {iterator} in range(self.D{offsets}[{collection_id}],"
                    f" self.D{offsets}[{collection_id} + 1]):"
                ),
                f"    {var} = self.D{flat}[{iterator}]",
                *[f"    {line}" for line in inner],
            ]

        if isinstance(collection, FixedSubset):
            iterator = self._iterator()
            skip = (
                [f"    if self.absent[{var}] != 0:", "        continue"]
                if collection.decision
                else []
            )
            return [
                (
                    f"for {iterator} in range(self.CS_off[{collection_id}],"
                    f" self.CS_off[{collection_id} + 1]):"
                ),
                f"    {var} = self.CS_flat[{iterator}]",
                *skip,
                *[f"    {line}" for line in inner],
            ]

        return [
            f"{var} = self.head[{collection_id}]",
            f"while {var} >= 0:",
            *[f"    {line}" for line in inner],
            f"    {var} = self.nxt[{var}]",
        ]

    def _iterator(self) -> str:
        """A fresh loop counter.

        Returns:
            Its generated name.
        """
        self.loop_count += 1
        return f"it{self.loop_count}_"

    def _data(self, data: NDArray[Any]) -> int:
        """The number of a data array, numbered the first time the generated code reads it.

        Args:
            data: The data.

        Returns:
            Its number.
        """
        for number, known in enumerate(self.registry):
            if known is data:
                return number

        self.registry.append(data)
        return len(self.registry) - 1

    def _const(self) -> str:
        """The generated reference to the next constant of the instance.

        Returns:
            The reference.
        """
        slot = self.next_const
        self.next_const += 1
        return (
            f"self.T{self.template.index}_P[inst * {len(self.template.shape.constants)} + {slot}]"
        )

    def _collection_ref(self, collection: object) -> str:
        """The reference to an aggregate's collection, in the order the shape numbered it.

        Args:
            collection: The collection.

        Returns:
            The generated reference to its id, or the index of a family read at a computed index.
        """
        base = collection.base if isinstance(collection, BoundFilteredSet) else collection

        if isinstance(base, ComputedSubset):
            return self._gen(base.key)

        slot = self.next_collection
        self.next_collection += 1
        stride = len(self.template.shape.collections)
        return f"self.T{self.template.index}_C[inst * {stride} + {slot}]"

    def _temp(self, prefix: str) -> str:
        """A fresh temporary.

        Args:
            prefix: What it holds (``ag``, ``ex``, ``v``…).

        Returns:
            Its generated name.
        """
        self.n_temps += 1
        return f"{prefix}{self.n_temps - 1}_"

    def _gen(self, expr: Expr) -> str:  # noqa: C901, PLR0911, PLR0912, PLR0914  per node class
        """The generated expression of a node; statements it needs go to ``self.preamble``.

        Args:
            expr: The node.

        Returns:
            The generated expression.

        Raises:
            ValueError: A node the generator does not know.
        """
        match expr:
            case Constant():
                return self._const()
            case Member(symbol=symbol):
                return self.env[symbol]
            case Position(offset=offset):
                return self.sequence_env[offset]
            case ValueRead(element=element):
                return f"self.start[{self._gen(element)}]"
            case TargetRead(element=element):
                return f"self.target[{self._gen(element)}]"
            case Lookup(data=data, index=index):
                return f"self.D{self._data(data)}[{self._gen(index)}]"
            case Arithmetic(operator="max" | "min" as operator, left=left, right=right):
                return f"{operator}({self._gen(left)}, {self._gen(right)})"
            case Arithmetic(operator=operator, left=left, right=right):
                return f"({self._gen(left)} {operator} {self._gen(right)})"
            case Comparison(
                operator="<", left=Constant(), right=Aggregate(operator="count") as count
            ) if self.next_const in self.template.zero_columns:
                return self._exists(count, negate=False, zero_first=True)
            case Comparison(
                operator="==", left=Aggregate(operator="count") as count, right=Constant()
            ) if self.next_const + self._count_consts(count) in self.template.zero_columns:
                return self._exists(count, negate=True, zero_first=False)
            case Comparison(operator=operator, left=left, right=right):
                return f"(1 if {self._gen(left)} {operator} {self._gen(right)} else 0)"
            case Logical(operator=operator, left=left, right=right):
                return self._lazy(operator, left, right)
            case Not(operand=operand):
                return f"(1 if ({self._gen(operand)}) == 0 else 0)"
            case IfThenElse(condition=condition, when_true=when_true, when_false=when_false):
                generated = self._gen(condition), self._gen(when_true), self._gen(when_false)
                return f"({generated[1]} if ({generated[0]}) != 0 else {generated[2]})"
            case Size(collection=collection):
                return self._size(collection)
            case Aggregate():
                return self._aggregate(expr)

        msg = f"node {type(expr).__name__}"
        raise ValueError(msg)

    def _size(self, collection: object) -> str:
        """The size of a collection, into a temporary.

        Args:
            collection: The collection.

        Returns:
            The temporary's name.
        """
        name = self._temp("sz")
        collection_id = self._collection_ref(collection)

        if isinstance(collection, PartitionPart):
            self.preamble.append(f"{name} = self.size[{collection_id}]")
        else:
            self.preamble += [
                f"{name} = 0",
                *self._iterate(collection, collection_id, self._temp("w"), [f"{name} += 1"]),
            ]

        return name

    def _aggregate(self, aggregate: Aggregate) -> str:
        """A count, sum, minimum or maximum over a collection, into a temporary.

        Args:
            aggregate: The aggregate.

        Returns:
            The temporary's name.
        """
        name, var = self._temp("ag"), self._temp("v")
        self.env[aggregate.member.symbol] = var
        collection_id = self._collection_ref(aggregate.member.collection)

        saved, self.preamble = self.preamble, []
        inner = self._gen(aggregate.body)
        inner_preamble, self.preamble = self.preamble, saved

        statement = {
            "count": f"{name} += 1 if ({inner}) != 0 else 0",
            "sum": f"{name} += {inner}",
            "min": f"{name} = min({name}, {inner})",
        }[aggregate.operator]
        initial = str(self.BIG) if aggregate.operator == "min" else "0"

        self.preamble += [
            f"{name} = {initial}",
            *self._iterate(
                aggregate.member.collection, collection_id, var, [*inner_preamble, statement]
            ),
        ]
        return name

    def _count_consts(self, expr: object) -> int:
        """How many constants a node (or a collection it ranges over) holds.

        Args:
            expr: The node or collection.

        Returns:
            The number of constants under it.
        """
        match expr:
            case Constant():
                return 1
            case Member(collection=collection) | Size(collection=collection):
                return self._count_consts(collection)
            case BoundFilteredSet(base=base):
                return self._count_consts(base)
            case ComputedSubset(key=key):
                return self._count_consts(key)
            case Expr():
                return sum(self._count_consts(child) for child in expr.children)

        return 0

    def _exists(self, count: Aggregate, *, negate: bool, zero_first: bool) -> str:
        """``0 < count(member, condition)`` or ``count(...) == 0``: a loop ending at a first match.

        Args:
            count: The count.
            negate: Whether it is compared ``== 0``.
            zero_first: Whether the 0 is written before the count (``0 < count``).

        Returns:
            The generated expression.
        """
        if zero_first:
            self._const()

        name, var = self._temp("ex"), self._temp("v")
        self.env[count.member.symbol] = var
        collection_id = self._collection_ref(count.member.collection)

        saved, self.preamble = self.preamble, []
        inner = self._gen(count.body)
        inner_preamble, self.preamble = self.preamble, saved

        if not zero_first:
            self._const()

        found = [*inner_preamble, f"if ({inner}) != 0:", f"    {name} = 1", "    break"]
        self.preamble += [
            f"{name} = 0",
            *self._iterate(count.member.collection, collection_id, var, found),
        ]
        return f"(1 if {name} == 0 else 0)" if negate else name

    def _lazy(self, operator: str, left: Expr, right: Expr) -> str:
        """Evaluate the right side only when it can change the result.

        Args:
            operator: ``and`` or ``or``.
            left: The left condition.
            right: The right condition.

        Returns:
            The generated expression.
        """
        left_code = self._gen(left)
        saved, self.preamble = self.preamble, []
        right_code = self._gen(right)
        right_preamble, self.preamble = self.preamble, saved

        if not right_preamble:
            return f"(1 if (({left_code}) != 0 {operator} ({right_code}) != 0) else 0)"

        name = self._temp("lz")
        self.preamble += [
            f"{name} = 1 if ({left_code}) != 0 else 0",
            f"if {name} {'!= 0' if operator == 'and' else '== 0'}:",
            *["    " + line for line in right_preamble],
            f"    {name} = 1 if ({right_code}) != 0 else 0",
        ]
        return name
