# PyML -- Python support vector machine experiment to predict atom's MM in a heirachical manner
-------

##### Pre-Alpha state, requires sklearn, matplotlib, numpy and optionally seaborn for beter graphics

##ChangeLog 25/8/16:
* **line 82--85:** Input training data edited to a more sensible order, each stage recursively uses the previous stage's input, this could now be moved into a function later.
* **line 93--130:** Implemented recursive learning structure and predictions are online, initial examination appears good, needs testing and validation against expected and against predictions just using geometry to validate the strength of including more data.
* Included analysis data showing "conservation of the sum of monopole magnitude", or charge conservation is obeyed and the model is truely self consistent at least for the monopole.


![Alt text](/Analysis/charge_conservation.png?raw=true "Charge is conserved")
