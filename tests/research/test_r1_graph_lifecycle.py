def test_frozen_plan_resident_execution_survives_graph_reset_boundaries():
    from freetoken.engine.graph import GraphRunner

    class Cache:
        def __init__(self, resident_only):
            self.resident_only = resident_only
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

    runner = GraphRunner.__new__(GraphRunner)
    resident = Cache(True)
    runner.moe_offload_cache = resident
    runner._reset_moe_offload_cache()
    runner._reset_moe_offload_cache()
    assert resident.reset_calls == 0

    ordinary = Cache(False)
    runner.moe_offload_cache = ordinary
    runner._reset_moe_offload_cache()
    assert ordinary.reset_calls == 1
