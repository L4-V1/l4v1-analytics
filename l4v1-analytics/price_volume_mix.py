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
    """
    Groups and aggregates two dataframes by specified columns and metrics.

    :param df: The primary dataframe for analysis.
    :param df_comparison: The comparison dataframe.
    :param group_by_columns: Columns to group by.
    :param volume_metric_name: The column name for volume metric.
    :param outcome_metric_name: The column name for outcome metric.
    :return: A tuple of LazyFrames after grouping and aggregation.
    """
    volume_expr = pl.col(volume_metric_name).sum().cast(pl.Float64)
    outcome_expr = pl.col(outcome_metric_name).sum().cast(pl.Float64)

    def group_df(df: pl.LazyFrame):
        return df.group_by(group_by_columns).agg(volume_expr, outcome_expr)

    return group_df(df), group_df(df_comparison)


def _get_join_key_expression(group_by_columns: List[str]) -> pl.Expr:
    """
    Generates a Polars expression to create a concatenated key from specified columns.

    :param group_by_columns: Columns to concatenate for a unique key.
    :return: A Polars expression that concatenates specified columns into a unique key.
    """
    group_keys = list()  # Create empty list to hold values

    for join_key in group_by_columns:  # Loop through join column keys
        temp_expr = (
            pl.when(pl.col(join_key).is_null())
            .then(pl.col(f"{join_key}_old"))  # If new key is null use old key
            .otherwise(join_key)  # Otherwise use new key
        ).str.to_lowercase()

        group_keys.append(temp_expr)  # Append results to the list

    # Concatenate unique values returned by each expression
    expr = pl.concat_str(*group_keys, separator="_").alias("group_keys")

    return expr


def _get_impact_expressions(
    volume_metric_name: str, outcome_metric_name: str
) -> Tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
    """
    Creates expressions to calculate various impact metrics between two datasets.

    :param volume_metric_name: Column name for volume data.
    :param outcome_metric_name: Column name for outcome data.
    :return: Tuple of expressions for volume, rate, mix, new, and old impacts.
    """

    # volume metric
    volume_new = pl.col(volume_metric_name)
    volume_old = pl.col(f"{volume_metric_name}_old")
    volume_diff = volume_new - volume_old

    # Outcome metric
    outcome_new = pl.col(outcome_metric_name)
    outcome_old = pl.col(f"{outcome_metric_name}_old")

    # Rate metric
    rate_new = outcome_new / volume_new
    rate_old = outcome_old / volume_old
    rate_avg_old = outcome_old.sum() / volume_old.sum()

    # Define impact expressions
    volume_impact = volume_diff * rate_avg_old
    mix_impact = (rate_old - rate_avg_old) * volume_diff
    rate_impact = (rate_new - rate_old) * volume_new

    def impact_expression(expr: pl.Expr, name: str) -> pl.Expr:
        expr = (
            pl.when((outcome_old.is_null()) | (outcome_new.is_null()))
            .then(pl.lit(0))
            .otherwise(expr)
        ).alias(f"{name}_impact")

        return expr

    volume_impact_expr = impact_expression(volume_impact, "volume")
    rate_impact_expr = impact_expression(rate_impact, "rate")
    mix_impact_expr = impact_expression(mix_impact, "mix")
    new_impact = (
        pl.when((outcome_old.is_null()) & (outcome_new.is_not_null()))
        .then(outcome_new)
        .otherwise(pl.lit(0))
        .alias("new_impact")
    )
    old_impact = (
        pl.when((outcome_new.is_null()) & (outcome_old.is_not_null()))
        .then((outcome_old * -1))
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
    df: Union[pl.LazyFrame, pl.DataFrame],
    df_comparison: Union[pl.LazyFrame, pl.DataFrame],
    group_by_columns: Union[str, List[str]],
    volume_metric_name: str,
    outcome_metric_name: str,
) -> pl.DataFrame:
    """
    Computes a DataFrame showing the impacts of volume, rate, mix, new, and old.

    :param df: Dataframe to analyze.
    :param df_comparison: Comparison dataframe.
    :param group_by_columns: Columns to group the data.
    :param volume_metric_name: Volume metric column name.
    :param outcome_metric_name: Outcome metric column name.
    :return: DataFrame with calculated impact metrics.
    """
    # Validate inputs
    if isinstance(group_by_columns, str):
        group_by_columns = [group_by_columns]
    if isinstance(df, pl.DataFrame):
        df = df.lazy()
    if isinstance(df_comparison, pl.DataFrame):
        df_comparison = df_comparison.lazy()

    df, df_comparison = _group_dataframes(
        df,
        df_comparison,
        group_by_columns,
        volume_metric_name,
        outcome_metric_name,
    )

    impact_expressions = _get_impact_expressions(
        volume_metric_name, outcome_metric_name
    )

    pvm_table = (
        df.join(df_comparison, how="outer", on=group_by_columns, suffix="_old")
        .select(
            _get_join_key_expression(group_by_columns),
            cs.numeric(),
            *impact_expressions,
        )
        .with_columns(cs.numeric().fill_nan(0).fill_null(0))
        .sort(by="group_keys")
    )

    return pvm_table.collect()


def _default_data_label_format(value: float) -> str:
    """
    Formats a float value to a string with commas as thousands separators.

    :param value: Float value to format.
    :return: Formatted string.
    """
    return f"{value:,.0f}"


def _create_data_label(
    value: float, previous_value: float, format_func: Callable
) -> str:
    """
    Creates a formatted data label showing the value and the change from a previous value.

    :param value: Current value.
    :param previous_value: Previous value to compare against.
    :param format_func: Function to format the float values.
    :return: Formatted data label string.
    """
    formatted_value = format_func(value)
    if previous_value is not None:
        growth = value - previous_value
        sign = "+" if growth >= 0 else ""
        formatted_growth = f"{sign}{format_func(growth)}"
        return f"{formatted_value} ({formatted_growth})"
    return formatted_value


def pvm_plot(
    pvm_table: pl.DataFrame,
    outcome_metric_name: str,
    primary_label: str = None,
    comparison_label: str = None,
    format_data_labels: Callable[[float], str] = None,
    plotly_params: Dict[str, Any] = {},
) -> go.Figure:
    """
    Generates a waterfall plot from provided performance measurement (PVM) data,
    illustrating the changes between current and comparison datasets.

    :param pvm_table: DataFrame containing the performance data. It should include
                      both current values and comparison values (historical, forecasted, etc.).
    :param outcome_metric_name: Name of the outcome metric column to be visualized in the plot.
                                This metric should exist as both current and historical or
                                comparative values in the `pvm_table`.
    :param primary_label: Custom label for the primary (current) data in the plot.
                          Defaults to the `outcome_metric_name` if not specified.
    :param comparison_label: Custom label for the comparison data in the plot.
                             Defaults to 'COMPARISON {outcome_metric_name}' if not specified.
    :param format_data_labels: A callable that formats the numerical data labels.
                               If None, defaults to a simple comma-separated formatter.
    :param plotly_params: A dictionary containing configuration options for Plotly,
                          such as graph dimensions and color schemes.

    :return: A Plotly Figure object representing the waterfall plot, which can be
             displayed or further modified.
    """
    # Check if formatting function is provided
    if format_data_labels is None:
        format_data_labels = _default_data_label_format
    if primary_label is None:
        primary_label = f"{outcome_metric_name}"
    if comparison_label is None:
        comparison_label = f"COMPARISON {outcome_metric_name}"

    # Initialize chart data
    x_labels, y_values, text_values, measure_list = [], [], [], []

    # Calculate old total
    outcome_old_x_label = f"<b>{comparison_label}</b>".upper()
    x_labels.append(outcome_old_x_label)
    outcome_old = pvm_table.get_column(f"{outcome_metric_name}_old").sum()
    y_values.append(outcome_old)
    text_values.append(
        f"<b>{_create_data_label(outcome_old, None, format_data_labels)}</b>"
    )
    measure_list.append("absolute")

    # Define impact types and cumulative sum start value
    impact_types = ["volume", "rate", "mix", "old", "new"]
    cumulative_sum = outcome_old
    previous_value = outcome_old

    # Loop through each impact type and add impacts
    for impact_type in impact_types:
        for key in pvm_table.get_column("group_keys").unique().sort(descending=True):
            impact_value = (
                pvm_table.filter(pl.col("group_keys") == key)
                .get_column(f"{impact_type}_impact")
                .sum()
            )
            if impact_value != 0:
                label = f"({impact_type[0]}.) {key}".lower()
                x_labels.append(label)
                y_values.append(impact_value)
                text_values.append(format_data_labels(impact_value))
                measure_list.append("relative")
                cumulative_sum += impact_value

        # Add subtotal for impact type
        subtotal_label = f"<b>{impact_type.capitalize()} Impact Subtotal</b>"
        x_labels.append(subtotal_label)
        y_values.append(cumulative_sum)
        text_values.append(
            _create_data_label(cumulative_sum, previous_value, format_data_labels)
        )
        measure_list.append("absolute")
        previous_value = cumulative_sum

    # Calculate "Outcome New" and add it to the chart
    outcome_new = pvm_table.get_column(outcome_metric_name).sum()
    outcome_new_label = f"<b>{primary_label}</b>".upper()
    x_labels.append(outcome_new_label)
    y_values.append(cumulative_sum)
    text_values.append(
        f"<b>{_create_data_label(outcome_new, outcome_old, format_data_labels)}</b>"
    )
    measure_list.append("total")

    # Extract trace settings if provided
    trace_settings = {
        "increasing": plotly_params.get("increasing", {"marker": {"color": "#00AF00"}}),
        "decreasing": plotly_params.get("decreasing", {"marker": {"color": "#E10000"}}),
        "totals": plotly_params.get(
            "totals",
            {"marker": {"color": "#F1F1F1", "line": {"color": "black", "width": 1}}},
        ),
    }

    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=measure_list,
            x=y_values,
            y=x_labels,
            text=text_values,
            textposition="auto",
            textfont=dict(size=8),
            **trace_settings,
        )
    )

    # Apply user-defined layout parameters if any
    layout_params = plotly_params.get(
        "layout", {"height": (len(x_labels) * 25 + 100), "template": "plotly_white"}
    )
    fig.update_layout(**layout_params)

    return fig
