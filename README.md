# groundwater_tools

A library to analyse groundwater survey and geophysical logging data and build simple analytical groundwater flow models.

## marinelli.py

This first iteration only contains the library **marinelli.py**, which contains classes
for constructing the simple analytical groundwater flow model by Marinelli & Niccoli
([2000](https://doi.org/10.1111/j.1745-6584.2000.tb00342.x)) for assessing flow into a
mine pit.

To construct a groundwater flow model, call the `PitFlow` class (all units in meters and
seconds) or `PitFlowCommonUnits` (units in more common formats).
To construct several models (e.g., for comparing various scenarios), use
`PitFlowCollection` or `PitflowCommonUnitsCollection`.

The classes contain methods to calculate inflow, water table depression, and to output
`matplotlib` depression cone profiles and `pandas.DataFrame` reports.
More exact documentation is work-in-progress.

Recommended to be used with Jupyter Notebooks.

## To do

* Add more testing for the critical mathematical methods.
* A more straightforward method to account for springtime high water. 
* Better code quality: type hints, docstrings, etc.
* Add proper documentation.