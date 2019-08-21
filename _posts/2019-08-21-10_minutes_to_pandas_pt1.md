---
title: 10 Minutes to Pandas, in R (Part One)
date: 2019-08-21
tags: ['tutorial']
---

Introduction
============

I started using R and the Tidyverse recently for EDA tasks recently. I
have a lot of experience cleaning data in Pandas, and when I started
using R I found myself often thinking “I know how to do this task in
pandas… I wish there was some type of thing that would show me how to
accomplish basic tasks that I do in pandas in R”. That’s why I made
this. I’ll be going through all sections of the [10 Minutes to
Pandas](https://pandas.pydata.org/pandas-docs/stable/10min.html)
tutorial, and showing how to accomplish these tasks in R (where
Relevant).

Imports
=======

Python
------

``` python
import numpy as np
import pandas as pd
```

R
-

``` r
knitr::opts_chunk$set(echo = TRUE)
library(reticulate)
library(tidyverse)
```

    ## ── Attaching packages ─────────────────────────────────────────────────────────────────────────────────────────────────────── tidyverse 1.2.1 ──

    ## ✔ ggplot2 3.2.1     ✔ purrr   0.3.2
    ## ✔ tibble  2.1.3     ✔ dplyr   0.8.3
    ## ✔ tidyr   0.8.3     ✔ stringr 1.4.0
    ## ✔ readr   1.3.1     ✔ forcats 0.4.0

    ## ── Conflicts ────────────────────────────────────────────────────────────────────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()

``` r
path_to_python <- "/home/jeb/anaconda3"
use_condaenv(path_to_python)
```

Object Creation
===============

Series – Vectors
----------------

The basic data structure that we work with in Pandas is a series, which
more or less represents a column in a dataframe. The analagous data
structure in R is called a vector.

In Pandas, we create a series by passing a list of values, and Pandas
makes a default integer index. We’ll talk more about indices later. Note
that NaN values represented by np.nan.

``` python
s = pd.Series([1,3,5,np.nan,6,8])
print(s)
```

    ## 0    1.0
    ## 1    3.0
    ## 2    5.0
    ## 3    NaN
    ## 4    6.0
    ## 5    8.0
    ## dtype: float64

In R, we accomplish the same thing with vectors. The assignment operator
is an arrow “&lt;-” pointing towards the name, rather than Python’s
equal sign. Notice that R’s NA value is built in. Like a Pandas Series,
an R vector can only contain one type. R has another data structure
called list() that allows mixed types, but we won’t be looking at that.

``` r
s <- c(1, 3, 5, NA, 6, 8)
print(s)
```

    ## [1]  1  3  5 NA  6  8

DataFrame() – Data.Frame and Tibble
-----------------------------------

There are a couple of ways we can make a DataFrame in Pandas. For
example, we can pass a Numpy array, with an datetime index and labeled
columns.

``` python
dates = pd.date_range('20130101', periods=6)
print(dates, "\n")
```

    ## (DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
    ##                '2013-01-05', '2013-01-06'],
    ##               dtype='datetime64[ns]', freq='D'), '\n')

``` python
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print(df)
```

    ##                    A         B         C         D
    ## 2013-01-01 -1.325551  0.193445  0.614884 -0.419290
    ## 2013-01-02 -1.322542 -1.150502 -0.328880  0.034351
    ## 2013-01-03  0.143935 -0.233053 -0.618928 -0.514150
    ## 2013-01-04  0.463793  0.984832 -0.698747 -0.564141
    ## 2013-01-05 -0.303595  0.793149 -0.775932 -0.596265
    ## 2013-01-06 -0.432535 -0.464467  0.371945  0.074324

This is one of the largest points of divergence for R and Pandas when it
comes to representing DataFrames. Pandas bases many things on the index,
which is different than a row name and is (supposed to, but not eforced
to be) a unique value denoting an observation. Base R has something
called rownames, which is similar to an index. Tibble, which is a type
of DataFrame used in the Tidyverse, does not use any type of index or
row name. So to make this dataframe in base R we would do the following:

``` r
library(lubridate) # Package within the tidyverse for dealing with dates
```

    ##
    ## Attaching package: 'lubridate'

    ## The following object is masked from 'package:base':
    ##
    ##     date

``` r
dates <- seq(ymd('2013-01-01'), ymd('2013-01-06'), by = '1 day')
print(dates)
```

    ## [1] "2013-01-01" "2013-01-02" "2013-01-03" "2013-01-04" "2013-01-05"
    ## [6] "2013-01-06"

``` r
df <- as.data.frame(
  replicate(4, rnorm(6)),
  row.names = as.character(dates),
)
colnames(df) <- c("A", "B", "C", "D")

print(df)
```

    ##                     A          B           C           D
    ## 2013-01-01  1.3517579  1.4891854 -0.74662792 -0.08712716
    ## 2013-01-02 -1.1477927 -0.7087421  0.04185842  0.34123797
    ## 2013-01-03 -2.3347115  0.0890571  2.07766702  0.76154220
    ## 2013-01-04 -1.7245894 -1.2985616 -0.47738833  0.27724998
    ## 2013-01-05  0.7136585 -0.2150484 -0.65781398  0.19012324
    ## 2013-01-06 -0.9650539  1.7198417  1.18187791 -0.10058367

In Pandas, we can also make a table by passing a dict of objects that
can be converted to series-like.

``` python
import pandas as pd
import numpy as np
df2 = pd.DataFrame({ 'A' : 1.,
                     'B' : pd.Timestamp('20130102'),
                     'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                     'D' : np.array([3] * 4,dtype='int32'),
                     'E' : pd.Categorical(["test","train","test","train"]),
                     'F' : 'foo' })
print(df2)
```

    ##      A          B    C  D      E    F
    ## 0  1.0 2013-01-02  1.0  3   test  foo
    ## 1  1.0 2013-01-02  1.0  3  train  foo
    ## 2  1.0 2013-01-02  1.0  3   test  foo
    ## 3  1.0 2013-01-02  1.0  3  train  foo

The columns of the resulting data frame have different dtypes:

``` python
print(df2.dtypes)
```

    ## A           float64
    ## B    datetime64[ns]
    ## C           float32
    ## D             int32
    ## E          category
    ## F            object
    ## dtype: object

We can make a tibble in a similar way. A tibble is a dataframe, but with
some extra features. It doesn’t automatically coerce character data into
factors (like data.frame does), and is more consistent with return types
in subsetting. Notice we don’t have to do anything else to check the
dtypes, as they’re displayed when we print the tibble. If your tibble
has too many columns to view them all just by printing, you can use
`glimpse(tibble)` From here on out we’ll be working only with tibbles in
R.

``` r
df2 <- tibble(
  A = 1,
  B = as_date("20130102"),
  C = rep(1.0, 4),
  D = rep(3, 4),
  E = factor(c("test", "train", "test", "train")),
  F = "foo"
)

print(df2)
```

    ## # A tibble: 4 x 6
    ##       A B              C     D E     F    
    ##   <dbl> <date>     <dbl> <dbl> <fct> <chr>
    ## 1     1 2013-01-02     1     3 test  foo  
    ## 2     1 2013-01-02     1     3 train foo  
    ## 3     1 2013-01-02     1     3 test  foo  
    ## 4     1 2013-01-02     1     3 train foo

*Note*: We can turn any data.frame into a tibble by using as\_tibble.
Just remember tibbles don’t have rownames, so we need to set them as a
column.

``` r
df <- as_tibble(df, rownames="Dates")
df
```

    ## # A tibble: 6 x 5
    ##   Dates           A       B       C       D
    ##   <chr>       <dbl>   <dbl>   <dbl>   <dbl>
    ## 1 2013-01-01  1.35   1.49   -0.747  -0.0871
    ## 2 2013-01-02 -1.15  -0.709   0.0419  0.341
    ## 3 2013-01-03 -2.33   0.0891  2.08    0.762
    ## 4 2013-01-04 -1.72  -1.30   -0.477   0.277
    ## 5 2013-01-05  0.714 -0.215  -0.658   0.190
    ## 6 2013-01-06 -0.965  1.72    1.18   -0.101

Viewing Data
============

In Python, you can get the top and bottom rows of the data frame:

``` python
print(df.head())
```

    ##                    A         B         C         D
    ## 2013-01-01 -1.325551  0.193445  0.614884 -0.419290
    ## 2013-01-02 -1.322542 -1.150502 -0.328880  0.034351
    ## 2013-01-03  0.143935 -0.233053 -0.618928 -0.514150
    ## 2013-01-04  0.463793  0.984832 -0.698747 -0.564141
    ## 2013-01-05 -0.303595  0.793149 -0.775932 -0.596265

``` python
print(df.tail(3))
```

    ##                    A         B         C         D
    ## 2013-01-04  0.463793  0.984832 -0.698747 -0.564141
    ## 2013-01-05 -0.303595  0.793149 -0.775932 -0.596265
    ## 2013-01-06 -0.432535 -0.464467  0.371945  0.074324

R is more or less the same:

``` r
head(df)
```

    ## # A tibble: 6 x 5
    ##   Dates           A       B       C       D
    ##   <chr>       <dbl>   <dbl>   <dbl>   <dbl>
    ## 1 2013-01-01  1.35   1.49   -0.747  -0.0871
    ## 2 2013-01-02 -1.15  -0.709   0.0419  0.341
    ## 3 2013-01-03 -2.33   0.0891  2.08    0.762
    ## 4 2013-01-04 -1.72  -1.30   -0.477   0.277
    ## 5 2013-01-05  0.714 -0.215  -0.658   0.190
    ## 6 2013-01-06 -0.965  1.72    1.18   -0.101

``` r
tail(df, 3)
```

    ## # A tibble: 3 x 5
    ##   Dates           A      B      C      D
    ##   <chr>       <dbl>  <dbl>  <dbl>  <dbl>
    ## 1 2013-01-04 -1.72  -1.30  -0.477  0.277
    ## 2 2013-01-05  0.714 -0.215 -0.658  0.190
    ## 3 2013-01-06 -0.965  1.72   1.18  -0.101

Summary Statistics
------------------

Pandas can show us some summary statistics of our data:

``` python
print(df.describe())
```

    ##               A         B         C         D
    ## count  6.000000  6.000000  6.000000  6.000000
    ## mean  -0.462749  0.020567 -0.239276 -0.330862
    ## std    0.739792  0.803438  0.592361  0.304575
    ## min   -1.325551 -1.150502 -0.775932 -0.596265
    ## 25%   -1.100040 -0.406614 -0.678792 -0.551643
    ## 50%   -0.368065 -0.019804 -0.473904 -0.466720
    ## 75%    0.032052  0.643223  0.196739 -0.079059
    ## max    0.463793  0.984832  0.614884  0.074324

Base R has a similar function called `summary()`, but a much better
function called `describe()` exists within the `psych` package. To
install it, you use `install.packages("psych")`. Then we can either load
the entire package by using `library(psych)`, or just use the functions
we need, as I show below:

``` r
psych::describe(df)
```

    ## Warning in psych::describe(df): NAs introduced by coercion

    ## Warning in FUN(newX[, i], ...): no non-missing arguments to min; returning
    ## Inf

    ## Warning in FUN(newX[, i], ...): no non-missing arguments to max; returning
    ## -Inf

    ##        vars n  mean   sd median trimmed  mad   min  max range skew
    ## Dates*    1 6   NaN   NA     NA     NaN   NA   Inf -Inf  -Inf   NA
    ## A         2 6 -0.68 1.43  -1.06   -0.68 1.44 -2.33 1.35  3.69 0.31
    ## B         3 6  0.18 1.20  -0.06    0.18 1.39 -1.30 1.72  3.02 0.18
    ## C         4 6  0.24 1.15  -0.22    0.24 0.72 -0.75 2.08  2.82 0.56
    ## D         5 6  0.23 0.32   0.23    0.23 0.32 -0.10 0.76  0.86 0.43
    ##        kurtosis   se
    ## Dates*       NA   NA
    ## A         -1.81 0.58
    ## B         -1.88 0.49
    ## C         -1.65 0.47
    ## D         -1.36 0.13

Transposing
-----------

We can transpose a DataFrame easily in python:

``` python
print(df.T)
```

    ##    2013-01-01  2013-01-02  2013-01-03  2013-01-04  2013-01-05  2013-01-06
    ## A   -1.325551   -1.322542    0.143935    0.463793   -0.303595   -0.432535
    ## B    0.193445   -1.150502   -0.233053    0.984832    0.793149   -0.464467
    ## C    0.614884   -0.328880   -0.618928   -0.698747   -0.775932    0.371945
    ## D   -0.419290    0.034351   -0.514150   -0.564141   -0.596265    0.074324

This is much more convoluted in R. This is partially because we aren’t
allowed to have indices in tibbles, and partially because the transpose
function in r outputs a matrix, which we then have to change back to a
tibble.

``` r
df_t <- as_tibble(t(select(df, -Dates)), rownames = "index")
```

    ## Warning: `as_tibble.matrix()` requires a matrix with column names or a `.name_repair` argument. Using compatibility `.name_repair`.
    ## This warning is displayed once per session.

``` r
colnames(df_t) <- c("index", df$Dates)
df_t
```

    ## # A tibble: 4 x 7
    ##   index `2013-01-01` `2013-01-02` `2013-01-03` `2013-01-04` `2013-01-05`
    ##   <chr>        <dbl>        <dbl>        <dbl>        <dbl>        <dbl>
    ## 1 A           1.35        -1.15        -2.33         -1.72         0.714
    ## 2 B           1.49        -0.709        0.0891       -1.30        -0.215
    ## 3 C          -0.747        0.0419       2.08         -0.477       -0.658
    ## 4 D          -0.0871       0.341        0.762         0.277        0.190
    ## # … with 1 more variable: `2013-01-06` <dbl>

Sorting
-------

In Pandas, we can sort our dataframe by columns (axis=1), or by rows
(asix=0):

``` python
print(df.sort_index(axis=1, ascending=False))
```

    ##                    D         C         B         A
    ## 2013-01-01 -0.419290  0.614884  0.193445 -1.325551
    ## 2013-01-02  0.034351 -0.328880 -1.150502 -1.322542
    ## 2013-01-03 -0.514150 -0.618928 -0.233053  0.143935
    ## 2013-01-04 -0.564141 -0.698747  0.984832  0.463793
    ## 2013-01-05 -0.596265 -0.775932  0.793149 -0.303595
    ## 2013-01-06  0.074324  0.371945 -0.464467 -0.432535

We can also sort by values of a column:

``` python
print(df.sort_values(by='B'))
```

    ##                    A         B         C         D
    ## 2013-01-02 -1.322542 -1.150502 -0.328880  0.034351
    ## 2013-01-06 -0.432535 -0.464467  0.371945  0.074324
    ## 2013-01-03  0.143935 -0.233053 -0.618928 -0.514150
    ## 2013-01-01 -1.325551  0.193445  0.614884 -0.419290
    ## 2013-01-05 -0.303595  0.793149 -0.775932 -0.596265
    ## 2013-01-04  0.463793  0.984832 -0.698747 -0.564141

We don’t need to worry about sorting by a row index in R for obvious
reasons, but we can sort columns.

``` r
select(df, order(names(df), decreasing=TRUE))
```

    ## # A tibble: 6 x 5
    ##   Dates            D       C       B      A
    ##   <chr>        <dbl>   <dbl>   <dbl>  <dbl>
    ## 1 2013-01-01 -0.0871 -0.747   1.49    1.35
    ## 2 2013-01-02  0.341   0.0419 -0.709  -1.15
    ## 3 2013-01-03  0.762   2.08    0.0891 -2.33
    ## 4 2013-01-04  0.277  -0.477  -1.30   -1.72
    ## 5 2013-01-05  0.190  -0.658  -0.215   0.714
    ## 6 2013-01-06 -0.101   1.18    1.72   -0.965

We can also sort on a column:

``` r
arrange(df, B)
```

    ## # A tibble: 6 x 5
    ##   Dates           A       B       C       D
    ##   <chr>       <dbl>   <dbl>   <dbl>   <dbl>
    ## 1 2013-01-04 -1.72  -1.30   -0.477   0.277
    ## 2 2013-01-02 -1.15  -0.709   0.0419  0.341
    ## 3 2013-01-05  0.714 -0.215  -0.658   0.190
    ## 4 2013-01-03 -2.33   0.0891  2.08    0.762
    ## 5 2013-01-01  1.35   1.49   -0.747  -0.0871
    ## 6 2013-01-06 -0.965  1.72    1.18   -0.101

Selection
=========

Getting
-------

Python: select a single column, which yields a `Series`:

``` python
print(df["A"])
```

    ## 2013-01-01   -1.325551
    ## 2013-01-02   -1.322542
    ## 2013-01-03    0.143935
    ## 2013-01-04    0.463793
    ## 2013-01-05   -0.303595
    ## 2013-01-06   -0.432535
    ## Freq: D, Name: A, dtype: float64

We can accomplish the same thing using `df.A`. You’ll notice that Pandas
quite often uses methods of DataFrame objects to manipulate them, while
R requires passing a tibble or data frame into a function.

R: Because we have no index, selection just gets us a vector. We can use
the base R method:

``` r
df$A
```

    ## [1]  1.3517579 -1.1477927 -2.3347115 -1.7245894  0.7136585 -0.9650539

or the Tidyverse method, using [dplyr](https://dplyr.tidyverse.org/).
This returns not a vector, but a tibble with a single column.:

``` r
dplyr::select(df, A)
```

    ## # A tibble: 6 x 1
    ##        A
    ##    <dbl>
    ## 1  1.35
    ## 2 -1.15
    ## 3 -2.33
    ## 4 -1.72
    ## 5  0.714
    ## 6 -0.965

Selection by Label
------------------

We select things by index using `loc` in Pandas. You can get more detail
on this in [10 Minutes to
Pandas](https://pandas.pydata.org/pandas-docs/stable/10min.html), but I
won’t address it here as there is no feature analagous to an index to
use for selection in R.

Selection by Position
---------------------

We select things by position in Python by using `iloc`. We can pass
numerical indices for both rows and columns.

``` python
print(df.iloc[3])
```

    ## A    0.463793
    ## B    0.984832
    ## C   -0.698747
    ## D   -0.564141
    ## Name: 2013-01-04 00:00:00, dtype: float64

By integer slices, acting similar to numpy/python:

``` python
print(df.iloc[3:5,0:2])
```

    ##                    A         B
    ## 2013-01-04  0.463793  0.984832
    ## 2013-01-05 -0.303595  0.793149

By lists of integer position locations, similar to the numpy/python
style:

``` python
print(df.iloc[[1,2,4],[0,2]])
```

    ##                    A         C
    ## 2013-01-02 -1.322542 -0.328880
    ## 2013-01-03  0.143935 -0.618928
    ## 2013-01-05 -0.303595 -0.775932

For slicing rows explicitly:

``` python
print(df.iloc[1:3,:])
```

    ##                    A         B         C         D
    ## 2013-01-02 -1.322542 -1.150502 -0.328880  0.034351
    ## 2013-01-03  0.143935 -0.233053 -0.618928 -0.514150

For slicing columns explicitly:

``` python
print(df.iloc[:,1:3])
```

    ##                    B         C
    ## 2013-01-01  0.193445  0.614884
    ## 2013-01-02 -1.150502 -0.328880
    ## 2013-01-03 -0.233053 -0.618928
    ## 2013-01-04  0.984832 -0.698747
    ## 2013-01-05  0.793149 -0.775932
    ## 2013-01-06 -0.464467  0.371945

For getting a value explicitly:

``` python
print(df.iloc[1,1])
```

    ## -1.15050198645

If we were to do this in R using the Tidyverse funcitons, we would need
to use `dplyr` functions `slice()`, which selects rows by position, and
`select()`, which is used to select columns either by name or position.
This time, its easier just to do things in base R. I won’t go through
all the examples but selection by position using `df[,]`, where row
positions are denoted before the comma, and column positions come after.

``` r
df[,1:3]
```

    ## # A tibble: 6 x 3
    ##   Dates           A       B
    ##   <chr>       <dbl>   <dbl>
    ## 1 2013-01-01  1.35   1.49  
    ## 2 2013-01-02 -1.15  -0.709
    ## 3 2013-01-03 -2.33   0.0891
    ## 4 2013-01-04 -1.72  -1.30  
    ## 5 2013-01-05  0.714 -0.215
    ## 6 2013-01-06 -0.965  1.72

Note that this selects different columns than the equivalent Pandas
method. This is for two reasons: - We’re selecting date as a column,
because Tibbles don’t use indices. - R starts indexing at 1, while
Python starts at 0.

We can use similar syntax to select rows by position, and columns by
name by passing a vector of column names:

``` r
df[1:4, c("Dates", "B", "D")]
```

    ## # A tibble: 4 x 3
    ##   Dates            B       D
    ##   <chr>        <dbl>   <dbl>
    ## 1 2013-01-01  1.49   -0.0871
    ## 2 2013-01-02 -0.709   0.341
    ## 3 2013-01-03  0.0891  0.762
    ## 4 2013-01-04 -1.30    0.277

Boolean Indexing
----------------

*Python*

Using a single column’s values to select data.

``` python
print(df[df.A > 0])
```

    ##                    A         B         C         D
    ## 2013-01-03  0.143935 -0.233053 -0.618928 -0.514150
    ## 2013-01-04  0.463793  0.984832 -0.698747 -0.564141

Selecting values from a DataFrame where a boolean condition is met:

``` python
print(df[df > 0])
```

    ##                    A         B         C         D
    ## 2013-01-01       NaN  0.193445  0.614884       NaN
    ## 2013-01-02       NaN       NaN       NaN  0.034351
    ## 2013-01-03  0.143935       NaN       NaN       NaN
    ## 2013-01-04  0.463793  0.984832       NaN       NaN
    ## 2013-01-05       NaN  0.793149       NaN       NaN
    ## 2013-01-06       NaN       NaN  0.371945  0.074324

Using the `isin()` method for filtering.

``` python
df2 = df.copy()

df2['E'] = ['one', 'one','two','three','four','three']

print(df2, "\n")
```

    ## (                   A         B         C         D      E
    ## 2013-01-01 -1.325551  0.193445  0.614884 -0.419290    one
    ## 2013-01-02 -1.322542 -1.150502 -0.328880  0.034351    one
    ## 2013-01-03  0.143935 -0.233053 -0.618928 -0.514150    two
    ## 2013-01-04  0.463793  0.984832 -0.698747 -0.564141  three
    ## 2013-01-05 -0.303595  0.793149 -0.775932 -0.596265   four
    ## 2013-01-06 -0.432535 -0.464467  0.371945  0.074324  three, '\n')

``` python
print(df2[df2['E'].isin(['two','four'])])
```

    ##                    A         B         C         D     E
    ## 2013-01-03  0.143935 -0.233053 -0.618928 -0.514150   two
    ## 2013-01-05 -0.303595  0.793149 -0.775932 -0.596265  four

*R*  
We use `dplyr` functions for this

``` r
df %>%
  filter(A > 0)
```

    ## # A tibble: 2 x 5
    ##   Dates          A      B      C       D
    ##   <chr>      <dbl>  <dbl>  <dbl>   <dbl>
    ## 1 2013-01-01 1.35   1.49  -0.747 -0.0871
    ## 2 2013-01-05 0.714 -0.215 -0.658  0.190

Notice the usage of “the pipe”, which is represented by `%>%`. This is
used to make R code more readable by avoiding storing every
transformation as a new object. It passes the output of whatever is
before it as the first argument to the next function called.

From the whole tibble (this particular task is easier with Base R):

``` r
df_copy <- cbind(df)
df_copy[df_copy < 0] <- NA
df_copy
```

    ##        Dates         A         B          C         D
    ## 1 2013-01-01 1.3517579 1.4891854         NA        NA
    ## 2 2013-01-02        NA        NA 0.04185842 0.3412380
    ## 3 2013-01-03        NA 0.0890571 2.07766702 0.7615422
    ## 4 2013-01-04        NA        NA         NA 0.2772500
    ## 5 2013-01-05 0.7136585        NA         NA 0.1901232
    ## 6 2013-01-06        NA 1.7198417 1.18187791        NA

Analagous to Pandas’ `isin()`

``` r
df_copy <- df_copy %>%
  mutate(E = c("one", "one", "two", "three", "four", "three"))

df_copy %>%
  filter(E %in% c("two", "four"))
```

    ##        Dates         A         B        C         D    E
    ## 1 2013-01-03        NA 0.0890571 2.077667 0.7615422  two
    ## 2 2013-01-05 0.7136585        NA       NA 0.1901232 four

Notice we used `dplyr`’s `mutate()` function to create the new variable
from a vector. Alternatively, we could have done it using base R like
this: `df$E <- c("one", "one", "two", "three", "four", "three")`.

Setting
-------

*Python*

Setting a new column automatically aligns the data by the indices.

``` python
s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))

df['F'] = s1
```

Setting values by label:

``` python
df.at[dates[0],'A'] = 0
```

Setting values by position:

``` python
df.iat[0,1] = 0
```

The result of the prior setting operations.

``` python
print(df)
```

    ##                    A         B         C         D    F
    ## 2013-01-01  0.000000  0.000000  0.614884 -0.419290  NaN
    ## 2013-01-02 -1.322542 -1.150502 -0.328880  0.034351  1.0
    ## 2013-01-03  0.143935 -0.233053 -0.618928 -0.514150  2.0
    ## 2013-01-04  0.463793  0.984832 -0.698747 -0.564141  3.0
    ## 2013-01-05 -0.303595  0.793149 -0.775932 -0.596265  4.0
    ## 2013-01-06 -0.432535 -0.464467  0.371945  0.074324  5.0

A where operation with setting.

``` python
df2 = df.copy()

df2[df2 > 0] = -df2

print(df2)
```

    ##                    A         B         C         D    F
    ## 2013-01-01  0.000000  0.000000 -0.614884 -0.419290  NaN
    ## 2013-01-02 -1.322542 -1.150502 -0.328880 -0.034351 -1.0
    ## 2013-01-03 -0.143935 -0.233053 -0.618928 -0.514150 -2.0
    ## 2013-01-04 -0.463793 -0.984832 -0.698747 -0.564141 -3.0
    ## 2013-01-05 -0.303595 -0.793149 -0.775932 -0.596265 -4.0
    ## 2013-01-06 -0.432535 -0.464467 -0.371945 -0.074324 -5.0

*R*

``` r
library(magrittr);
```

    ##
    ## Attaching package: 'magrittr'

    ## The following object is masked from 'package:purrr':
    ##
    ##     set_names

    ## The following object is masked from 'package:tidyr':
    ##
    ##     extract

``` r
s1 = c(1, 2, 3, 4, 5, 6)

df %<>% mutate(F = s1)

df
```

    ## # A tibble: 6 x 6
    ##   Dates           A       B       C       D     F
    ##   <chr>       <dbl>   <dbl>   <dbl>   <dbl> <dbl>
    ## 1 2013-01-01  1.35   1.49   -0.747  -0.0871     1
    ## 2 2013-01-02 -1.15  -0.709   0.0419  0.341      2
    ## 3 2013-01-03 -2.33   0.0891  2.08    0.762      3
    ## 4 2013-01-04 -1.72  -1.30   -0.477   0.277      4
    ## 5 2013-01-05  0.714 -0.215  -0.658   0.190      5
    ## 6 2013-01-06 -0.965  1.72    1.18   -0.101      6

Since we don’t have indices in R, we instead have to rely on having a
vector with the same length as our tibble, and add the column using the
`mutate()` function. To clear up any confusion, the `%<>%` adds the
column F to df in place, and is equivalent to the statement
`df <- df %>% mutate(F=s1)`. You can see how we’re slightly mismatched
from the pandas version because of the dates index mismatch. To do this
correctly, we’d have to use a join (that’s coming up later).

``` r
df[df$Dates == dates[1], 'A'] <- 0
```

``` r
df[1, 3] <- 0
print(df)
```

    ## # A tibble: 6 x 6
    ##   Dates           A       B       C       D     F
    ##   <chr>       <dbl>   <dbl>   <dbl>   <dbl> <dbl>
    ## 1 2013-01-01  0      0      -0.747  -0.0871     1
    ## 2 2013-01-02 -1.15  -0.709   0.0419  0.341      2
    ## 3 2013-01-03 -2.33   0.0891  2.08    0.762      3
    ## 4 2013-01-04 -1.72  -1.30   -0.477   0.277      4
    ## 5 2013-01-05  0.714 -0.215  -0.658   0.190      5
    ## 6 2013-01-06 -0.965  1.72    1.18   -0.101      6

This time we have to increment the column by two rather than one,
because Dates counts as a column.

``` r
#df2 = df.copy()
#df2[df2 > 0] = -df2
#print(df2)


df2 = cbind(df) %>%
  select(-Dates)

df2[df2 > 0] <-  -df2[df2 > 0]

# df2 <- cbind(df$Dates, df2)
df2 %<>% mutate(Dates = df$Dates)
print(df2)
```

    ##            A          B           C           D  F      Dates
    ## 1  0.0000000  0.0000000 -0.74662792 -0.08712716 -1 2013-01-01
    ## 2 -1.1477927 -0.7087421 -0.04185842 -0.34123797 -2 2013-01-02
    ## 3 -2.3347115 -0.0890571 -2.07766702 -0.76154220 -3 2013-01-03
    ## 4 -1.7245894 -1.2985616 -0.47738833 -0.27724998 -4 2013-01-04
    ## 5 -0.7136585 -0.2150484 -0.65781398 -0.19012324 -5 2013-01-05
    ## 6 -0.9650539 -1.7198417 -1.18187791 -0.10058367 -6 2013-01-06

You’ll see that to do a lot of these setting operations, its easier to
work with Base R than the Tidyverse. Also things that use indices in
Pandas can be a little inconvenient, because they get treated like any
other column in R. In the final case we have to remove the Dates column
and then put it back in at the end.

Missing Data
============

*Python*

Pandas primarily uses the value np.nan to represent missing data. It is
by default not included in computations. Reindexing allows you to
change/add/delete the index on a specified axis. This returns a copy of
the data.

``` python
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])

df1.loc[dates[0]:dates[1], 'E'] = 1

print(df1)
```

    ##                    A         B         C         D    F    E
    ## 2013-01-01  0.000000  0.000000  0.614884 -0.419290  NaN  1.0
    ## 2013-01-02 -1.322542 -1.150502 -0.328880  0.034351  1.0  1.0
    ## 2013-01-03  0.143935 -0.233053 -0.618928 -0.514150  2.0  NaN
    ## 2013-01-04  0.463793  0.984832 -0.698747 -0.564141  3.0  NaN

To drop any rows that have missing data.

``` python
print(df1.dropna(how='any'))
```

    ##                    A         B        C         D    F    E
    ## 2013-01-02 -1.322542 -1.150502 -0.32888  0.034351  1.0  1.0

Filling missing data.

``` python
print(df1.fillna(value=5))
```

    ##                    A         B         C         D    F    E
    ## 2013-01-01  0.000000  0.000000  0.614884 -0.419290  5.0  1.0
    ## 2013-01-02 -1.322542 -1.150502 -0.328880  0.034351  1.0  1.0
    ## 2013-01-03  0.143935 -0.233053 -0.618928 -0.514150  2.0  5.0
    ## 2013-01-04  0.463793  0.984832 -0.698747 -0.564141  3.0  5.0

To get the boolean mask where values are nan.

``` python
print(pd.isna(df1))
```

    ##                 A      B      C      D      F      E
    ## 2013-01-01  False  False  False  False   True  False
    ## 2013-01-02  False  False  False  False  False  False
    ## 2013-01-03  False  False  False  False  False   True
    ## 2013-01-04  False  False  False  False  False   True

*R*

``` r
df1 <- as_tibble(py$df1, rownames= 'Dates')
```

I got tired of trying to make these pandas dataframes from scratch in R,
so I used the
[reticulate](https://cran.r-project.org/web/packages/reticulate/vignettes/r_markdown.html" class="uri">https://cran.r-project.org/web/packages/reticulate/vignettes/r_markdown.html)
package to pass our python dataframe directly to an R dataframe, and
then converted it to a tibble.

To drop any rows that have missing data:

``` r
print(drop_na(df1))
```

    ## # A tibble: 1 x 7
    ##   Dates          A     B      C      D     F     E
    ##   <chr>      <dbl> <dbl>  <dbl>  <dbl> <dbl> <dbl>
    ## 1 2013-01-02 -1.32 -1.15 -0.329 0.0344     1     1

``` r
df1_copy <- cbind(df1)

df1_copy[is.na(df1_copy)] <- 5
print(df1_copy)
```

    ##        Dates          A          B          C          D F E
    ## 1 2013-01-01  0.0000000  0.0000000  0.6148837 -0.4192899 5 1
    ## 2 2013-01-02 -1.3225421 -1.1505020 -0.3288799  0.0343515 1 1
    ## 3 2013-01-03  0.1439346 -0.2330534 -0.6189276 -0.5141503 2 5
    ## 4 2013-01-04  0.4637933  0.9848324 -0.6987471 -0.5641412 3 5

Here we need to make sure to make a copy before filling the setting the
NA values to 5, otherwise we will alter our original dataframe. Pandas
offers an argument to create a new dataframe or do the operation in
place (with parameter `inplace=True`), but base R doesn’t have this
option. The tidyverse has a function to replace NA values called
`replace_na()`, but it is not meant to replace values in an entire
dataframe at once.

``` r
is.na(df1)
```

    ##      Dates     A     B     C     D     F     E
    ## [1,] FALSE FALSE FALSE FALSE FALSE  TRUE FALSE
    ## [2,] FALSE FALSE FALSE FALSE FALSE FALSE FALSE
    ## [3,] FALSE FALSE FALSE FALSE FALSE FALSE  TRUE
    ## [4,] FALSE FALSE FALSE FALSE FALSE FALSE  TRUE

Thanks for reading, and I hope this was helpful! I’ll be covering the
second half of the tutorial in another blog post.
