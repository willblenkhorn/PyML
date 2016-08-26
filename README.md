# PyML -- Python support vector machine experiment to predict atom's MM in a heirachical manner
-------

##### Pre-Alpha state
###### Requires sklearn, matplotlib, numpy and optionally seaborn for beter graphics

##ChangeLog 25/8/16:
* **line 82--85:** Input training data edited to a more sensible order, each stage recursively uses the previous stage's input, this could now be moved into a function later.
* **line 93--130:** Implemented recursive learning structure and predictions are online, initial examination appears good, needs testing and validation against expected and against predictions just using geometry to validate the strength of including more data.
* Included analysis reference data and graph showing charge conservation is obeyed in the calculated and predicted values. This is the first step to showing that the model is truely self consistent, working for at least the monopole.

* New model using geometry and previously predicted moments, behaves well/
![Alt text](/Analysis/charge_conservation.png?raw=true "Charge is conserved")
* Old model simplying using the geometry, behaves poorly.
![Alt text](/Analysis/charge_conservation_not_observed_old.png?raw=true "Old method charge conservation is poor")

* Prediction model looks good by early indications, it matches the calculated trends overall.
![Alt text](/Analysis/H2 predicted.png?raw=true "O1 vs H2 is well predicted")
![Alt text](/Analysis/H3 predicted.png?raw=true "O1 vs H3 is well predicted")
![Alt text](/Analysis/O1_q00_vs_geom.png?raw=true "O1 vs bondlen1 is well predicted")
* A 14% increase in performance is achieved by including the prior prediction of the oxygen in the model for the H2 atom.
![Alt text](/Analysis/cumulative_H2_q00_error.png?raw=true "New vs old method")

* Need to look at the H3 atom's monopole, then start moving onto dipoles.
