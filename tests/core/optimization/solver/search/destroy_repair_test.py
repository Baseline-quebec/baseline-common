from numpy.random import default_rng

from baseline_core.core.optimization.solver.binding.model_compiler import ModelCompiler
from baseline_core.core.optimization.solver.search.destroy_repair import DestroyRepair, Release
from baseline_core.core.optimization.solver.search.kernel_state import KernelState
from baseline_core.core.optimization.solver.search.search_index import SearchIndex
from tests.core.optimization.job_shop_model import EarlyStartModelDefinition, Shop


class TestInOrder:
    def test__a_released_tail__is_re_placed_in_order_each_at_its_earliest_free_start(self) -> None:
        """Three operations spread out on M1: the last two close up behind the first, in order."""
        shop = Shop.of([[name, f"J{name}", 0, 2, ["M1"], 0, 40] for name in ["a", "b", "c"]])
        bound = ModelCompiler(EarlyStartModelDefinition(), cythonize=False).bind(shop)
        kernel = bound.kernel
        kernel.load(start=[0, 10, 20], machine=[0, 0, 0])
        kernel.soft_mode = True
        release = Release([1, 2], remove_first=True, in_order=True)

        DestroyRepair(release, SearchIndex.of(bound), default_rng(0)).apply(KernelState(kernel))

        assert kernel.placement == [[0, 2, 4], [0, 0, 0]]
        assert kernel.full_hard == 0

    def test__a_released_tail__keeps_the_order_it_had(self) -> None:
        """Released last-first, the elements still land in the order of their former starts."""
        shop = Shop.of([[name, f"J{name}", 0, 2, ["M1"], 0, 40] for name in ["a", "b", "c"]])
        bound = ModelCompiler(EarlyStartModelDefinition(), cythonize=False).bind(shop)
        kernel = bound.kernel
        kernel.load(start=[0, 20, 10], machine=[0, 0, 0])
        kernel.soft_mode = True
        release = Release([2, 1], remove_first=True, in_order=True)

        DestroyRepair(release, SearchIndex.of(bound), default_rng(0)).apply(KernelState(kernel))

        assert kernel.placement[0] == [0, 4, 2]
