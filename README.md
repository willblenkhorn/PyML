# PyML -- Python support vector machine experiment to predict atom's MM in a heirachical manner
-------

##### Pre-Alpha state
###### Requires sklearn, matplotlib, numpy and optionally seaborn for better graphics.

## News:
* Centred and scaled data, drastically improving accuracy from prototype by 100x.

* SVR model which includes oxygen monopole drastically reduces prediction error for the H2 and H3 atoms by 44% and 52%, respectively, when compared to just using input geometry alone. To note a graphs further to the left indicate an improved error. 
![Figure1](/Analysis/cumulative_H2 standard scaled new vs old model_error.png?raw=true "Hydrogen2 monopole prediction improved")
Average error reduced by: 44%
![Figure2](/Analysis/cumulative_H3 standard scaled new vs old model_error.png?raw=true "Hydrogen3 monopole prediction improved")
Average error reduced by: 52%
