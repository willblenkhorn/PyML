# PyML
## A Python support vector machine experiment to predict atomic multipole moments in a hierarchical manner

###### Requires sklearn, matplotlib, numpy and optionally seaborn for better graphics.
## Setup 
### Using Conda
* Download Python 3.X conda variant from: http://conda.pydata.org/miniconda.html
* Once installed and in a new terminal run: **conda install scikit-learn matplotlib numpy seaborn scipy=0.17.1**

## News:
* Centred and scaled data, drastically improving accuracy from prototype by 100x.

* SVR model which includes oxygen monopole drastically reduces prediction error for the H2 and H3 atoms by 44% and 52%, respectively, when compared to just using input geometry alone. To note a graph further to the left indicates an improved error. 
![Figure1](/Analysis/cumulative_H2 standard scaled new vs old model_error.png?raw=true "Hydrogen2 monopole prediction improved")
Average error reduced by: 44%
![Figure2](/Analysis/cumulative_H3 standard scaled new vs old model_error.png?raw=true "Hydrogen3 monopole prediction improved")
Average error reduced by: 52%
