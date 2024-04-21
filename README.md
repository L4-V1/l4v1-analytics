# l4v1

l4v1 is a Python library designed to simplify some data-analytics tasks using data manipulation and visualization techniques. Built on top of Polars and Plotly, it offers a straightforward API for creating detailed summaries in a quick way. It is work in progress and more functionality is to be added in future.

## Installation

You can install the l4v1 package directly from PyPI:

```bash
pip install l4v1
```
## Usage

### Impact Analysis
#### Impact Table
The impact_table function allows you to perform comparative analysis between two datasets—your primary dataset and a comparison dataset. This function is versatile, suitable for various analyses including period-to-period comparisons, budget vs. actual evaluations, or any scenario where understanding the drivers of change in an outcome metric (like revenue, orders, or any quantifiable metric) is crucial.

By specifying group dimensions, a volume metric (e.g., sales units or website visits), and an outcome metric (e.g., revenue, orders), you can dissect the contributing factors to performance variations. This tool is invaluable for identifying why certain metrics have increased or decreased compared to another period or benchmark.

Here's an example of how to use impact_table to compare week-to-week sales data, focusing on differences across product categories:

```python
import polars as pl
from l4v1 import impact_table

# Load your datasets
sales_week1 = pl.read_csv("data/sales_week1.csv")
sales_week2 = pl.read_csv("data/sales_week2.csv")

# Perform the impact analysis
# Note, the primary and comparison DFs must be in the same format
impact_df = impact_table(
    df_primary=sales_week2, # Data to analyse
    df_comparison=sales_week1, # Data to compare against
    group_by_columns=["product_category"], # Dimension(s) to use
    volume_metric_name="item_quantity", # Column name containing volume (e.g. quantity)
    outcome_metric_name="revenue" # Column name containing outcome (e.g. revenue)
)

```
#### Impact Plot
After generating an impact table, you can visualize the results with impact_plot. This function creates a waterfall plot that highlights how different groups contributed to the overall change in outcomes:
```python
from l4v1 import impact_plot

# Visualize the impact analysis
fig = impact_plot(
    impact_table=impact_df,
    format_data_labels="{:,.0f}€", # Optional data label format, e.g. 1050.123 >> 1,050€
    primary_total_label="revenue week 2", # Optional label
    comparison_total_label="Revenue Week 1", # Optional label
    title="Impact Analysis Example", # Optional title
    color_total="lightgray", # Optional for total bar colors
    color_increase="#00AF00", # Optional for increase bar color
    color_decrease="#FF0000" # Optional for decrease bar color
)
fig.show()
```
This will generate a waterfall plot that illustrates how different product categories impacted in less sales in the week 2.

![Impact Plot Example](docs/impact_plot_example.png)

#### Interpreting the Results
The impact plot visualizes always three types of impacts rate, volume, and mix impact:

* Rate Impact: Reflects changes in average values within each category, such as average unit price.
* Volume Impact: Represents changes in quantity, like the number of units sold.
* Mix Impact: Indicates shifts in the distribution across categories, such as a larger proportion of sales from high-value items.

Additionally, if there are new or discontinued elements between the datasets, they are categorized under "new" or "old" impacts, signifying their presence in only one of the datasets.

The impact plot aids in understanding the drivers behind revenue changes between the compared datasets.

### Features
* Data Grouping: Group your data based on one or more columns.
* Impact Calculation: Automatically calculate volume, outcome, and rate impacts between two datasets.
* Visual Representation: Create waterfall plots to visually represent the impact analysis results.
