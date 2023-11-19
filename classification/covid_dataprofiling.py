from numpy import ndarray, log
from pandas import read_csv, DataFrame, Series, concat, get_dummies
from scipy.stats import norm, expon, lognorm
from seaborn import heatmap, pairplot

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, savefig, show, subplots

from sklearn.preprocessing import OneHotEncoder

from dslab_functions import plot_multiline_chart, plot_multi_scatters_chart, plot_bar_chart, plot_multibar_chart, set_chart_labels, get_variable_types, define_grid, HEIGHT, encode_cyclic_variables, dummify, determine_outlier_thresholds_for_var, count_outliers

# read file
filename = "class_pos_covid.csv"
file_tag = "class_pos_covid"
df : DataFrame = read_csv('class_pos_covid.csv')
### DATA DIMENSIONALITY ###

# nr of records vs nr of variables

figure(figsize=(4, 2))
values: dict[str, int] = {"nr records": df.shape[0], "nr variables": df.shape[1]}

plot_bar_chart(list(values.keys()), list(values.values()), title="Nr of records vs nr of variables")
savefig(f"images/{file_tag}_records_variables.png", bbox_inches = "tight")
show()

# description
df.describe(include="all")

# shape (Nr records x Nr variables)
print("Nr of records: ", len(df))
print("Dimensionality: ", df.shape)

# variable types
variable_types: dict[str, list] = get_variable_types(df)
print(variable_types)
counts: dict[str, int] = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])

figure(figsize=(4, 2))
plot_bar_chart(
    list(counts.keys()), list(counts.values()), title="Nr of variables per type"
)
savefig(f"images/{file_tag}_variable_types.png", bbox_inches = "tight")
show()

symbolic: list[str] = variable_types["symbolic"]
df[symbolic] = df[symbolic].apply(lambda x: x.astype("category"))
df.dtypes

# missing values
mv: dict[str, int] = {}
for var in df.columns:
    nr: int = df[var].isna().sum()
    if nr > 0:
        mv[var] = nr

figure(figsize = (18,5))
plot_bar_chart(
    list(mv.keys()),
    list(mv.values()),
    title="Nr of missing values per variable",
    xlabel="variables",
    ylabel="nr missing values",
)
savefig(f"images/{file_tag}_mv.png", bbox_inches = "tight")
show()

### DATA GRANULARITY ###

# symbolic variables

def analyse_property_granularity(
    data: DataFrame, property: str, vars: list[str]
) -> ndarray:
    cols: int = len(vars)
    fig: Figure
    axs: ndarray
    fig, axs = subplots(1, cols, figsize=(cols * HEIGHT * 4, HEIGHT), squeeze=False)
    fig.suptitle(f"Granularity study for {property}")
    for i in range(cols):
        counts: Series[int] = data[vars[i]].value_counts()
        plot_bar_chart(
            counts.index.to_list(),
            counts.to_list(),
            ax=axs[0, i],
            title=vars[i],
            xlabel=vars[i],
            ylabel="nr records",
            percentage=False,
        )
    return axs

prop1 = "location"
prop2 = "smoking"
analyse_property_granularity(df, prop1, ["State"])
savefig(f"images/{file_tag}_granularity_{prop1}.png", bbox_inches = "tight")
analyse_property_granularity(df, prop2, ["SmokerStatus", "ECigaretteUsage"])
savefig(f"images/{file_tag}_granularity_{prop2}.png", bbox_inches = "tight")
show()


### DATA DISTRIBUTION ###


# global boxplot
variables_types: dict[str, list] = get_variable_types(df)
numeric: list[str] = variables_types["numeric"]
if [] != numeric:
    df[numeric].boxplot(rot=90)
    savefig(f"images/{file_tag}_global_boxplot.png", bbox_inches = "tight")
    show()
else:
    print("There are no numeric variables.")
    
# boxplots for individual numeric vars
if [] != numeric:
    rows: int
    cols: int
    rows, cols = define_grid(len(numeric))
    fig: Figure
    axs: ndarray
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i, j = 0, 0
    for n in range(len(numeric)):
        axs[i, j].set_title("Boxplot for %s" % numeric[n])
        axs[i, j].boxplot(df[numeric[n]].dropna().values)
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    savefig(f"images/{file_tag}_single_boxplots.png")
    show()
else:
    print("There are no numeric variables.")

# standard outliers

if [] != numeric:
    outliers: dict[str, int] = count_outliers(df, numeric)
    figure(figsize=(12, HEIGHT))
    plot_multibar_chart(
        numeric,
        outliers,
        title="Nr of standard outliers per variable",
        xlabel="variables",
        ylabel="nr outliers",
        percentage=False,
    )
    savefig(f"images/{file_tag}_outliers_standard.png")
    show()
else:
    print("There are no numeric variables.")
    
    
# outliers
if [] != numeric:
    outliers: dict[str, int] = count_outliers(df, numeric, nrstdev=4, iqrfactor=4.5)
    figure(figsize=(12, HEIGHT))
    plot_multibar_chart(
        numeric,
        outliers,
        title="Nr of outliers per variable",
        xlabel="variables",
        ylabel="nr outliers",
        percentage=False,
    )
    savefig(f"images/{file_tag}_outliers.png")
    show()
else:
    print("There are no numeric variables.")
    

# histograms for numeric
if [] != numeric:
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i: int
    j: int
    i, j = 0, 0
    for n in range(len(numeric)):
        set_chart_labels(
            axs[i, j],
            title=f"Histogram for {numeric[n]}",
            xlabel=numeric[n],
            ylabel="nr records",
        )
        axs[i, j].hist(df[numeric[n]].dropna().values, "auto")
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    savefig(f"images/{file_tag}_single_histograms_numeric.png")
    show()
else:
    print("There are no numeric variables.")
    

# distributions for numeric


# histograms for symbolic
variables_types: dict[str, list] = get_variable_types(df)
symbolic: list[str] = variables_types["symbolic"] + variables_types["binary"]
if [] != symbolic:
    rows, cols = define_grid(len(symbolic))
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT * 2.5, rows * HEIGHT * 3.5), squeeze=False
    )
    i, j = 0, 0
    for n in range(len(symbolic)):
        counts: Series = df[symbolic[n]].value_counts()
        plot_bar_chart(
            counts.index.to_list(),
            counts.to_list(),
            ax=axs[i, j],
            title="Histogram for %s" % symbolic[n],
            xlabel=symbolic[n],
            ylabel="nr records",
            percentage=False,
        )
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    savefig(f"images/{file_tag}_histograms_symbolic.png", bbox_inches = "tight")
    show()
else:
    print("There are no symbolic variables.")
    

# class distribution
target = "CovidPos"

values: Series = df[target].value_counts()
print(values)

figure(figsize=(4, 2))
plot_bar_chart(
    values.index.to_list(),
    values.to_list(),
    title=f"Target distribution (target={target})",
)
savefig(f"images/{file_tag}_class_distribution.png", bbox_inches = "tight")
show()

### DATA SPARSITY ###

# scatter-plots (all x all - including class)


# correlation (all x all - including class)
variables_types: dict[str, list] = get_variable_types(df)
symbolic: list[str] = variables_types["symbolic"] + variables_types["binary"]

df2: DataFrame = dummify(df, symbolic)

variables_types2: dict[str, list] = get_variable_types(df2)
numeric2: list[str] = variables_types2["numeric"]
symbolic2: list[str] = variables_types2["symbolic"] + variables_types2["binary"]
all_vars = numeric2 + symbolic2

corr_mtx: DataFrame = df2[all_vars].corr().abs()

plt.figure(figsize=(12, 12))
heatmap(
    abs(corr_mtx),
    xticklabels=all_vars,
    yticklabels=all_vars,
    annot=False,
    cmap="Blues",
    vmin=0,
    vmax=1,
)

plt.xticks(fontsize=3)
plt.yticks(fontsize=3)

plt.savefig(f"images/{file_tag}_correlation_analysis.png", bbox_inches = "tight")
plt.show()