import polars as pl
import polars.selectors as cs
import plotly.graph_objects as go
from typing import Any, Callable, Dict, List, Tuple, Union


def _group_dataframes(
    df: Union[pl.LazyFrame, pl.DataFrame],
    df_comparison: Union[pl.LazyFrame, pl.DataFrame],
    group_by_columns: Union[str, List[str]],
    volume_metric_name: str,
    outcome_metric_name: str,
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    volume_expr = pl.col(volume_metric_name).sum().cast(pl.Float64)
    outcome_expr = pl.col(outcome_metric_name).sum().cast(pl.Float64)

    def group_df(df: pl.LazyFrame):
        return df.group_by(group_by_columns).agg(volume_expr, outcome_expr)

    return group_df(df), group_df(df_comparison)


def _get_join_key_expression(group_by_columns: List[str]) -> pl.Expr:
    group_keys = list()

    for join_key in group_by_columns:
        temp_expr = (
            pl.when(pl.col(join_key).is_null())
            .then(pl.col(f"{join_key}_comparison"))
            .otherwise(join_key)
        ).str.to_lowercase()

        group_keys.append(temp_expr)

    # Concatenate unique values returned by each expression
    expr = pl.concat_str(*group_keys, separator="_").alias("group_keys")

    return expr


def _get_impact_expressions(
    volume_metric_name: str,
    outcome_metric_name: str,
) -> Tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
    # Volume
    volume_new = pl.col(volume_metric_name)
    volume_comparison = pl.col(f"{volume_metric_name}_comparison")
    volume_diff = volume_new - volume_comparison

    # Outcome
    outcome_new = pl.col(outcome_metric_name)
    outcome_comparison = pl.col(f"{outcome_metric_name}_comparison")

    # Rate
    rate_new = outcome_new / volume_new
    rate_comparison = outcome_comparison / volume_comparison
    rate_avg_comparison = outcome_comparison.sum() / volume_comparison.sum()

    # Impact Expressions
    volume_impact = volume_diff * rate_avg_comparison
    mix_impact = (rate_comparison - rate_avg_comparison) * volume_diff
    rate_impact = (rate_new - rate_comparison) * volume_new

    def impact_expression(expr: pl.Expr, name: str) -> pl.Expr:
        expr = (
            pl.when((outcome_comparison.is_null()) | (outcome_new.is_null()))
            .then(pl.lit(0))
            .otherwise(expr)
        ).alias(f"{name}_impact")

        return expr

    volume_impact_expr = impact_expression(volume_impact, "volume")
    rate_impact_expr = impact_expression(rate_impact, "rate")
    mix_impact_expr = impact_expression(mix_impact, "mix")

    new_impact = (
        pl.when((outcome_comparison.is_null()) & (outcome_new.is_not_null()))
        .then(outcome_new)
        .otherwise(pl.lit(0))
        .alias("new_impact")
    )
    old_impact = (
        pl.when((outcome_new.is_null()) & (outcome_comparison.is_not_null()))
        .then((outcome_comparison * -1))
        .otherwise(pl.lit(0))
        .alias("old_impact")
    )

    return (
        volume_impact_expr,
        rate_impact_expr,
        mix_impact_expr,
        new_impact,
        old_impact,
    )


def pvm_table(
    df_primary: Union[pl.LazyFrame, pl.DataFrame],
    df_comparison: Union[pl.LazyFrame, pl.DataFrame],
    group_by_columns: Union[str, List[str]],
    volume_metric_name: str,
    outcome_metric_name: str,
) -> pl.DataFrame:
    if isinstance(group_by_columns, str):
        group_by_columns = [group_by_columns]
    if isinstance(df_primary, pl.DataFrame):
        df_primary = df_primary.lazy()
    if isinstance(df_comparison, pl.DataFrame):
        df_comparison = df_comparison.lazy()

    df_primary, df_comparison = _group_dataframes(
        df_primary,
        df_comparison,
        group_by_columns,
        volume_metric_name,
        outcome_metric_name,
    )

    impact_expressions = _get_impact_expressions(
        volume_metric_name, outcome_metric_name
    )

    pvm_table = (
        df_primary.join(
            df_comparison, how="outer", on=group_by_columns, suffix="_comparison"
        )
        .select(
            _get_join_key_expression(group_by_columns),
            cs.numeric(),
            *impact_expressions,
        )
        .with_columns(cs.numeric().fill_nan(0).fill_null(0))
        .sort(by="group_keys")
    )

    return pvm_table.collect()


def _create_data_label(
    value: float, previous_value: float, format_func: Callable
) -> str:
    formatted_value = format_func(value)
    if previous_value is not None:
        growth = value - previous_value
        sign = "+" if growth >= 0 else ""
        formatted_growth = f"{sign}{format_func(growth)}"
        return f"{formatted_value} ({formatted_growth})"
    return formatted_value


def _default_pvm_plot_settings(num_labels: int, user_settings: Dict[str, Any] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generates default trace and layout settings for plotly graphs and merges them with user provided settings."""
    trace_defaults = {
        "increasing": {"marker": {"color": "#00AF00"}},
        "decreasing": {"marker": {"color": "#E10000"}},
        "totals": {
            "marker": {"color": "#F1F1F1", "line": {"color": "black", "width": 1}}
        },
    }
    layout_defaults = {
        "height": num_labels * 25 + 100,
        "width": 750,
        "template": "plotly_white",
    }

    if user_settings:
        trace_settings = {**trace_defaults, **user_settings.get("trace_settings", {})}
        layout_settings = {**layout_defaults, **user_settings.get("layout", {})}
    else:
        trace_settings = trace_defaults
        layout_settings = layout_defaults

    return trace_settings, layout_settings


def pvm_plot(
    pvm_table: pl.DataFrame,
    outcome_metric_name: str,
    primary_label: str = None,
    comparison_label: str = None,
    format_data_labels: Callable[[float], str] = None,
    plotly_params: Dict[str, Any] = {},
) -> go.Figure:
    if format_data_labels is None:
        format_data_labels = lambda value: f"{value:,.0f}"
    primary_label = primary_label or outcome_metric_name
    comparison_label = comparison_label or f"COMPARISON {outcome_metric_name}"

    x_labels, y_values, data_labels, measure_list = [], [], [], []
    outcome_comparison = pvm_table.get_column(f"{outcome_metric_name}_comparison").sum()

    x_labels.append(f"<b>{comparison_label}</b>".upper())
    y_values.append(outcome_comparison)
    data_labels.append(f"<b>{format_data_labels(outcome_comparison)}</b>")
    measure_list.append("absolute")

    cumulative_sum = outcome_comparison
    previous_value = outcome_comparison

    impact_types = ["volume", "rate", "mix", "old", "new"]
    for impact_type in impact_types:
        for key in pvm_table.get_column("group_keys").unique().sort(descending=True):
            impact_value = (
                pvm_table.filter(pl.col("group_keys") == key)
                .get_column(f"{impact_type}_impact")
                .sum()
            )
            if impact_value != 0:
                x_labels.append(f"({impact_type[0]}.) {key}".lower())
                y_values.append(impact_value)
                data_labels.append(format_data_labels(impact_value))
                measure_list.append("relative")
                cumulative_sum += impact_value

        x_labels.append(f"<b>{impact_type.capitalize()} Impact Subtotal</b>")
        y_values.append(cumulative_sum)
        data_labels.append(
            _create_data_label(cumulative_sum, previous_value, format_data_labels)
        )
        measure_list.append("absolute")
        previous_value = cumulative_sum

    outcome_new = pvm_table.get_column(outcome_metric_name).sum()
    x_labels.append(f"<b>{primary_label}</b>".upper())
    y_values.append(outcome_new)
    data_labels.append(
        _create_data_label(outcome_new, outcome_comparison, format_data_labels)
    )
    measure_list.append("total")

    # Get PVM plot default settings
    trace_settings, layout_params = _default_pvm_plot_settings(len(x_labels), plotly_params)

    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=measure_list,
            x=y_values,
            y=x_labels,
            text=data_labels,
            textposition="auto",
            textfont=dict(size=8),
            **trace_settings,
        )
    )
    fig.update_layout(**layout_params)

    return fig
