from contextlib import ExitStack
from dataclasses import dataclass
from typing import List

import click
import gradio as gr
import pandas as pd

from parse_results import build_results


@dataclass
class PlotConfig:
    x_title: str
    y_title: str
    title: str
    percentiles: List[float] = None


def run(from_results_dir, datasource, port):
    css = '''
    .summary span {
        font-size: 10px;
        padding-top:0;
        padding-bottom:0;
    }
    '''

    df_bench = pd.DataFrame()
    line_plots_bench = []

    default_df = pd.DataFrame.from_dict({
        "rate": [1, 2],
        "inter_token_latency_ms_p90": [10, 20],
        "version": ["default", "default"],
        "run_id": ["default", "default"]
    })

    def load_demo(percentiles):
        return update_bench(percentiles)

    def update_bench(percentiles):
        res = []
        for plot in line_plots_bench:
            if plot['config'].percentiles:
                k = plot['metric'] + '_' + str(percentiles)
                df_bench[plot['metric']] = df_bench[k] if k in df_bench.columns else 0
            res.append(df_bench)
        return res

    def select_region(selection: gr.SelectData):
        min_w, max_w = selection.index
        data = df_bench[(df_bench['rate'] >= min_w) & (df_bench['rate'] <= max_w)]
        res = []
        for plot in line_plots_bench:
            metric = plot["metric"]
            y_min = data[metric].min()
            y_max = data[metric].max()
            res.append(gr.LinePlot(x_lim=[min_w, max_w], y_lim=[y_min, y_max]))
        return res

    def reset_region():
        return [gr.LinePlot(x_lim=None, y_lim=None) for _ in line_plots_bench]

    def load_datasource(datasource, fn):
        if datasource.startswith('file://') or datasource.startswith('s3://'):
            return fn(datasource)
        else:
            raise ValueError(f"Unknown datasource: {datasource}")

    def load_bench_results(source) -> pd.DataFrame:
        data = pd.read_parquet(source)
        data = data[(data['id'] != 'warmup') & (data['id'] != 'throughput')]
        data = data[data['executor_type'] == 'ConstantArrivalRate']
        return data

    if from_results_dir is not None:
        build_results(from_results_dir, 'benchmarks.parquet', None)

    df_bench = load_datasource(datasource, load_bench_results)

    # Rename for clarity
    df_bench = df_bench.rename(columns={
        'total_tokens': 'total_output_tokens',
        'total_tokens_sent': 'total_input_tokens'
    })

    metrics = {
        "inter_token_latency_ms": PlotConfig("QPS", "Time (ms)", "Inter Token Latency (lower is better)", [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]),
        "time_to_first_token_ms": PlotConfig("QPS", "Time (ms)", "TTFT (lower is better)", [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]),
        "e2e_latency_ms": PlotConfig("QPS", "Time (ms)", "End to End Latency (lower is better)", [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]),
        "token_throughput_secs": PlotConfig("QPS", "Tokens/s", "Request Output Throughput (higher is better)"),
        "successful_requests": PlotConfig("QPS", "Count", "Successful requests (higher is better)"),
        "error_rate": PlotConfig("QPS", "%", "Error rate"),
        "total_input_tokens": PlotConfig("QPS", "Count", "Total Input Tokens (higher is better)"),
        "total_output_tokens": PlotConfig("QPS", "Count", "Total Output Tokens (higher is better)")
    }

    # ✅ Deduplicate percentiles
    percentiles_set = set()
    for v in metrics.values():
        if v.percentiles:
            percentiles_set.update(f'p{int(p * 100)}' for p in v.percentiles)
    percentiles = sorted(percentiles_set)
    percentiles.append('avg')

    with gr.Blocks(css=css, title="Inference Benchmarker") as demo:
        with gr.Row():
            gr.Markdown("# Inference-benchmarker 🤗\n## Benchmarks results")

        with gr.Row():
            percentiles_bench = gr.Radio(percentiles, label="Select percentile", value="avg")

        i = 0
        with ExitStack() as stack:
            for k, v in metrics.items():
                if i % 2 == 0:
                    stack.close()
                    gs = stack.enter_context(gr.Row())
                line_plots_bench.append({
                    "component": gr.LinePlot(default_df, label=v.title, x="rate", y=k,
                                             y_title=v.y_title, x_title=v.x_title, color="run_id"),
                    "metric": k,
                    "config": v
                })
                i += 1

        percentiles_bench.change(update_bench, [percentiles_bench],
                                 [plot["component"] for plot in line_plots_bench])

        gr.on([plot["component"].select for plot in line_plots_bench],
              select_region, [],
              outputs=[plot["component"] for plot in line_plots_bench])

        gr.on([plot["component"].double_click for plot in line_plots_bench],
              reset_region, None,
              outputs=[plot["component"] for plot in line_plots_bench])

        demo.load(load_demo, [percentiles_bench],
                  [plot["component"] for plot in line_plots_bench])

    demo.launch(server_port=port, server_name="0.0.0.0")


@click.command()
@click.option('--from-results-dir', default=None, help='Load inference-benchmarker results from a directory')
@click.option('--datasource', default='file://benchmarks.parquet', help='Load a Parquet file already generated')
@click.option('--port', default=7860, help='Port to run the dashboard')
def main(from_results_dir, datasource, port):
    run(from_results_dir, datasource, port)


if __name__ == '__main__':
    main(auto_envvar_prefix='DASHBOARD')
