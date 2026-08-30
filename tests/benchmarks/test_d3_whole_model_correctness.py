from inferswarm_d3.whole_model_correctness import OUTPUT_CAP, PRIMARY, WORKER_A, WORKER_B, command, first_difference, token_hash


def test_local_command_is_gpu0_graph_only(tmp_path):
    got = command(tmp_path, "model", 1234, "local", "/placement.json")
    assert ["--gpu", PRIMARY] == got[got.index("--gpu"):got.index("--gpu") + 2]
    assert "--inferswarm-experimental-d3-graph-multiworker" not in got
    assert "--inferswarm-secondary-gpu" not in got
    assert "--inferswarm-experimental-d2-graph-remote" not in got
    assert got[got.index("--cuda-graph-max-bs") + 1] == "1"
    assert got[got.index("--moe-cache-size") + 1] == "3774"
    assert got[got.index("--num-tokens") + 1] == "17075"


def test_d3_commands_only_bind_active_uuid_workers(tmp_path):
    a = command(tmp_path, "model", 1, "a", "/p")
    b = command(tmp_path, "model", 1, "b", "/p")
    ab = command(tmp_path, "model", 1, "ab", "/p")
    assert WORKER_A in a and WORKER_B not in a
    assert WORKER_B in b and WORKER_A not in b
    assert WORKER_A in ab and WORKER_B in ab


def test_token_hash_is_compact_json_and_difference_is_exact():
    assert token_hash([1, 23]) == token_hash([1, 23])
    assert first_difference([1, 2], [1, 3]) == 1
    assert first_difference([1], [1, 2]) == 1
    assert first_difference([1, 2], [1, 2]) is None
    assert OUTPUT_CAP == 32
