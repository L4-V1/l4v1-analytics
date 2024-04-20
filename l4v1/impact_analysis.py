import polars as pl
import polars.selectors as cs
import plotly.graph_objects as go
from typing import Any, Callable, Dict, List, Tuple, Union, Optional


def _group_dataframe(
    lf: pl.LazyFrame,
    group_by_columns: List[str],
    metric_names: List[str],
) -> pl.LazyFrame:
    transformed_cols = [
        pl.col(col_name).cast(pl.Utf8).alias(col_name) for col_name in group_by_columns
    ]
    agg_expressions = [pl.col(metric).sum().cast(pl.Float64) for metric in metric_names]
    return lf.group_by(transformed_cols).agg(agg_expressions)


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
    expr = pl.concat_str(*group_keys, separator="|").alias("group_keys")

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
    rate_impact = (rate_new - rate_comparison) * volume_new
    volume_impact = volume_diff * rate_avg_comparison
    mix_impact = (rate_comparison - rate_avg_comparison) * volume_diff

    def impact_expression(expr: pl.Expr, name: str) -> pl.Expr:
        expr = (
            pl.when((outcome_comparison.is_null()) | (outcome_new.is_null()))
            .then(pl.lit(0))
            .otherwise(expr)
        ).alias(f"{name}_impact")

        return expr

    rate_impact_expr = impact_expression(rate_impact, "rate")
    volume_impact_expr = impact_expression(volume_impact, "volume")
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
        rate_impact_expr,
        volume_impact_expr,
        mix_impact_expr,
        new_impact,
        old_impact,
    )


def impact_table(
    df_primary: Union[pl.LazyFrame, pl.DataFrame],
    df_comparison: Union[pl.LazyFrame, pl.DataFrame],
    group_by_columns: Union[str, List[str]],
    volume_metric_name: str,
    outcome_metric_name: str,
) -> pl.DataFrame:
    # Ensure polars df type and convert to lazy
    if not all(
        isinstance(item, (pl.LazyFrame, pl.DataFrame))
        for item in (df_primary, df_comparison)
    ):
        raise TypeError(
            "df_primary and df_comparison must be Polars LazyFrame or DataFrame"
        )
    if isinstance(df_primary, pl.DataFrame):
        df_primary = df_primary.lazy()
    if isinstance(df_comparison, pl.DataFrame):
        df_comparison = df_comparison.lazy()

    # Validate group_by_columns
    if isinstance(group_by_columns, str):
        group_by_columns = [group_by_columns]
    elif isinstance(group_by_columns, list):
        if not all(isinstance(col, str) for col in group_by_columns):
            raise ValueError("All elements in group_by_columns must be strings.")
    else:
        raise TypeError(
            "group_by_columns must be a list of strings or a single string."
        )

    # Check metric column names are string types
    if not all(
        isinstance(item, str) for item in (volume_metric_name, outcome_metric_name)
    ):
        raise TypeError("volume_metric_name and outcome_metric_name must be strings.")

    df_primary, df_comparison = [
        _group_dataframe(
            df,
            group_by_columns,
            [volume_metric_name, outcome_metric_name],
        )
        for df in [df_primary, df_comparison]
    ]

    impact_expressions = _get_impact_expressions(
        volume_metric_name, outcome_metric_name
    )

    impact_table = (
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

    return impact_table.collect()


def _parse_metric_column_names(impact_table: pl.LazyFrame) -> tuple:
    metric_columns = impact_table.select(cs.ends_with("_comparison")).columns
    volume_col = metric_columns[0]
    outcome_col = metric_columns[1]

    volume_col_stripped = (
        volume_col.replace("_comparison", "")
        if volume_col.endswith("_comparison")
        else ValueError()
    )
    outcome_col_stripped = (
        outcome_col.replace("_comparison", "")
        if outcome_col.endswith("_comparison")
        else ValueError()
    )

    return volume_col_stripped, outcome_col_stripped


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


def _prep_data_for_impact_plot(
    impact_table: pl.DataFrame,
    format_data_labels: Callable,
    primary_total_label: str,
    comparison_total_label: str,
) -> Tuple[list, list, list, list]:
    _, outcome_metric_name = _parse_metric_column_names(impact_table)
    if format_data_labels is None:
        format_data_labels = lambda value: f"{value:,.0f}"
    primary_total_label = primary_total_label or outcome_metric_name
    comparison_total_label = (
        comparison_total_label or f"COMPARISON {outcome_metric_name}"
    )

    x_labels, y_values, data_labels, measure_list = [], [], [], []
    outcome_comparison = impact_table.get_column(
        f"{outcome_metric_name}_comparison"
    ).sum()

    x_labels.append(f"<b>{comparison_total_label}</b>".upper())
    y_values.append(outcome_comparison)
    data_labels.append(f"<b>{format_data_labels(outcome_comparison)}</b>")
    measure_list.append("absolute")

    cumulative_sum = outcome_comparison
    previous_value = outcome_comparison

    impact_types = ["rate", "volume", "mix", "old", "new"]
    if (impact_table.get_column("old_impact").sum() == 0) & (
        impact_table.get_column("new_impact").sum() == 0
    ):
        impact_types = ["rate", "volume", "mix"]

    for impact_type in impact_types:
        for key in impact_table.get_column("group_keys").unique().sort(descending=True):
            impact_value = (
                impact_table.filter(pl.col("group_keys") == key)
                .get_column(f"{impact_type}_impact")
                .sum()
            )
            # if impact_value != 0:
            x_labels.append(f"{key} ({impact_type[0]}.)".lower())
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

    outcome_new = impact_table.get_column(outcome_metric_name).sum()
    x_labels.append(f"<b>{primary_total_label}</b>".upper())
    y_values.append(outcome_new)
    data_labels.append(
        f"<b>{_create_data_label(outcome_new, outcome_comparison, format_data_labels)}</b>"
    )
    measure_list.append("total")

    return x_labels, y_values, data_labels, measure_list


def impact_plot(
    impact_table: pl.DataFrame,
    primary_total_label: Optional[str] = None,
    comparison_total_label: Optional[str] = None,
    format_data_labels: Optional[str] = "{:,.0f}",
    title: Optional[str] = None,
    color_increase: Optional[str] = "#00AF00",
    color_decrease: Optional[str] = "#FF0000",
    color_total: Optional[str] = "#F1F1F1",
    text_font_size: Optional[int] = 8,
    plot_height: Optional[int] = None,
    plot_width: Optional[int] = 750,
    plotly_template: Optional[str] = "plotly_white",
    plotly_trace_settings: Optional[Dict[str, Any]] = None,
    plotly_layout_settings: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    if not isinstance(impact_table, pl.DataFrame):
        raise TypeError("impact_table must be Polars DataFrame")

    # Check for columns containing "_comparison" in their names
    comparison_columns = [col for col in impact_table.columns if "_comparison" in col]
    if not comparison_columns:
        raise ValueError(
            "No comparison column found in the impact_table. Use impact_table function and use it's returned dataframe for plotting."
        )

    # Convert format string to a formatting function
    formatter = lambda x: format_data_labels.format(x)

    # Prepare data for plotting
    x_labels, y_values, data_labels, measure_list = _prep_data_for_impact_plot(
        impact_table, formatter, primary_total_label, comparison_total_label
    )

    # Create the plot
    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=measure_list,
            x=y_values,
            y=x_labels,
            text=data_labels,
            textposition="auto",
            textfont=dict(size=text_font_size),
            increasing=dict(marker=dict(color=color_increase)),
            decreasing=dict(marker=dict(color=color_decrease)),
            totals=dict(
                marker=dict(color=color_total, line=dict(color="black", width=1))
            ),
        )
    )

    # Update layout with basic settings
    layout_params = {
        "title": title,
        "height": plot_height if plot_height else len(x_labels) * 25 + 100,
        "width": plot_width,
        "template": plotly_template,
    }

    # Apply advanced settings if provided
    if plotly_trace_settings:
        fig.update_traces(plotly_trace_settings)
    if plotly_layout_settings:
        fig.update_layout(**plotly_layout_settings)
    else:
        fig.update_layout(**layout_params)

    return fig
