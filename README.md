# PyML -- Python support vector machine experiment to predict atom's MM in a heirachical manner
-------

##### Pre-Alpha state, requires sklearn, matplotlib, numpy and optionally seaborn for beter graphics

##ChangeLog:
* **line 82--85:** Input training data edited to a more sensible order, each stage recursively uses the previous stage's input, this could now be moved into a function later.
* **line 93--130:** Implemented recursive learning structure
