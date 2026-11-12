import pytest
from pytest_mock import MockerFixture

from baseline_core.core.optimization.solver.binding.model_compiler import ModelCompiler
from baseline_core.core.optimization.solver.infeasibility.precheck_run import PrecheckRun
from tests.core.optimization.job_shop_model import JobShopModelDefinition, Shop


class TestRead:
    @pytest.mark.parametrize("cythonize", [False, True])
    def test__each_lone_domain__is_read_once_and_blamed_once(
        self, mocker: MockerFixture, *, cythonize: bool
    ) -> None:
        """Starts 0 to 37 on M1 or M2: the window removes 32 starts, the machine option M2."""
        shop = Shop.of([["a", "J", 0, 3, ["M1"], 2, 10], ["b", "K", 0, 2, ["M1", "M2"], 0, 40]])
        bound = ModelCompiler(JobShopModelDefinition(), cythonize=cythonize).bind(shop)
        domain = mocker.spy(PrecheckRun, "domain")

        run = PrecheckRun.read(bound, [])

        assert domain.call_count == 2
        assert run.filtered.tolist() == [[32, 1, 0, 0], [0, 0, 0, 0]]
        assert (run.starts[0], run.sets[0]) == ([2, 3, 4, 5, 6, 7], [0])
