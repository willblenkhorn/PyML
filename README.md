# PyML -- Python support vector machine experiment to predict atom's MM in a heirachical manner
-------

##### Pre-Alpha state
###### Requires sklearn, matplotlib, numpy and optionally seaborn for beter graphics

##ChangeLog 25/8/16:
* **line 82--85:** Input training data edited to a more sensible order, each stage recursively uses the previous stage's input, this could now be moved into a function later.
* **line 93--130:** Implemented recursive learning structure and predictions are online, initial examination appears good, needs testing and validation against expected and against predictions just using geometry to validate the strength of including more data.
* Included analysis reference data and graph showing charge conservation is obeyed in the calculated and predicted values. This is the first step to showing that the model is truely self consistent, working for at least the monopole.


![Alt text](/Analysis/charge_conservation.png?raw=true "Charge is conserved")

* Prediction model looks good by early indications, it matches the calculated trends overall.
![Alt text](/Analysis/H2 predicted.png?raw=true "O1 vs H2 is well predicted")
![Alt text](/Analysis/H3 predicted.png?raw=true "O1 vs H3 is well predicted")
