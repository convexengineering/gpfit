## Plot unit tests

Plot unit tests are run with [pytest-mpl](https://github.com/matplotlib/pytest-mpl)

To generate the baseline images:
`pytest --mpl-generate-path=baseline t_plot_fit.py`

To test against the baseline images:
`pytest --mpl t_plot_fit.py`
 
