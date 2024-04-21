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
The impact_table function allows you to compare two datasets: a primary dataset and a comparison dataset. This function is ideal for analyzing changes over time or between different data segments. You specify the dimensions to group the data, a volume metric (such as sales units or website visitors), and an outcome metric (such as revenue or orders).

Here's how to compare sales data between two weeks, analyzing differences in product categories:

```python
import polars as pl
from l4v1 import impact_table

# Load your datasets
sales_week1 = pl.read_csv("data/sales_week1.csv")
sales_week2 = pl.read_csv("data/sales_week2.csv")

# Perform the impact analysis
impact_df = impact_table(
    sales_week2,
    sales_week1,
    group_by_columns=["product_category"],
    volume_metric_name="item_quantity",
    outcome_metric_name="revenue"
)

# Print the first few rows of the result
print(impact_df.head())

```
#### Impact Plot
After generating an impact table, you can visualize the results with impact_plot. This function creates a waterfall plot that highlights how different groups contributed to the overall change in outcomes:
```python
from l4v1 import impact_plot

# Visualize the impact analysis
fig = impact_plot(
    impact_table=impact_table_df,
    format_data_labels="{:,.0f}€", # Optional
    primary_total_label="REVENUE WEEK2", # Optional
    comparison_total_label="REVENUE WEEK1", # Optional
    title="Impact Analysis Example", # Optional
)
fig.show()
```
This will generate a waterfall plot that illustrates how different product categories impacted in less sales in the week 2.

![Impact Plot Example](docs/impact_plot_example.png)

#### Interpreting the Results
The impact plot visualizes three types of impacts:

* Rate Impact: Changes in the average rate value within each category (e.g., average unit price).
* Volume Impact: Changes in volume (e.g., quantities sold).
* Mix Impact: Effects due to the shift in proportions among categories (e.g., if a higher proportion of sales comes from high-value items).

### Features
Data Grouping: Group your data based on one or more columns.
Impact Calculation: Automatically calculate volume, outcome, and rate impacts between two datasets.
Visual Representation: Create waterfall plots to visually represent the impact analysis results.
