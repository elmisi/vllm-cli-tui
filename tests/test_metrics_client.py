"""The Prometheus text parser and the rate derivation."""
from vllm_tui.metrics_client import EndpointSample, derive_rates, parse_prometheus

SAMPLE = """\
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="my-model"} 4.0
vllm:num_requests_waiting{engine="0",model_name="my-model"} 2.0
vllm:gpu_cache_usage_perc{engine="0",model_name="my-model"} 0.37
vllm:prompt_tokens_total{engine="0",model_name="my-model"} 123456.0
vllm:generation_tokens_total{engine="0",model_name="my-model"} 7890.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="my-model"} 100.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="my-model"} 3.0
"""


def test_parses_gauges_and_counters():
    metrics = parse_prometheus(SAMPLE)
    assert metrics["vllm:num_requests_running"] == 4.0
    assert metrics["vllm:num_requests_waiting"] == 2.0
    assert metrics["vllm:gpu_cache_usage_perc"] == 0.37
    assert metrics["vllm:generation_tokens_total"] == 7890.0


def test_metrics_with_the_same_name_are_summed_across_label_sets():
    # request_success_total appears once per finished_reason
    metrics = parse_prometheus(SAMPLE)
    assert metrics["vllm:request_success_total"] == 103.0


def test_comments_and_garbage_lines_are_ignored():
    metrics = parse_prometheus("# just a comment\nnot a metric at all\n")
    assert metrics == {}


def test_token_rates_come_from_diffing_two_samples():
    before = EndpointSample(at=100.0, metrics={"vllm:generation_tokens_total": 1000.0,
                                               "vllm:prompt_tokens_total": 5000.0})
    after = EndpointSample(at=110.0, metrics={"vllm:generation_tokens_total": 1300.0,
                                              "vllm:prompt_tokens_total": 6000.0})
    rates = derive_rates(before, after)
    assert rates["generation_tok_s"] == 30.0
    assert rates["prompt_tok_s"] == 100.0


def test_a_server_restart_never_yields_negative_rates():
    before = EndpointSample(at=100.0, metrics={"vllm:generation_tokens_total": 9000.0})
    after = EndpointSample(at=110.0, metrics={"vllm:generation_tokens_total": 50.0})
    assert derive_rates(before, after)["generation_tok_s"] == 0.0


def test_rates_need_time_to_pass():
    s = EndpointSample(at=100.0, metrics={"vllm:generation_tokens_total": 1.0})
    assert derive_rates(s, s) == {}
