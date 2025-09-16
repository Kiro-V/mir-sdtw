Remove Report Title at the Cover Page

Remove the weight observation and class imbalance distribution (move to appendix)

Remove other learning curve (moving to appendix maybe)

Remove the predicted alignment (it does not make any sense)

Fix scale of alignment example (it started at -0.5 and then at end-0.5, which is weird)

Transpose the gradient matrix (it should have strong alignment at x-axis and the path move from bottom-left to top-right) so that it is intuitively meaningful

parameter: origin:'lower'

Table 4.1 - fix caption
fix epoch to convergence for all soft cases (they are currently shows total epoch trained)

Possible question - discussion: why all of them have the same F1 score?

Chords in axis y: reduce the amount of chord labeled on graph for cleaning look
Increase font of graphs

Explain how did we obtained the ground truth alignment for the gradient analysis graph.

Now that the ground truth alignment is a rectangle (any alignment within the ground truth box is correct), estimate it to be a diagonal line is better than the vertical/horizontal line.

The comparison between CTC and SDTW is optional

Possible Appendix Experiment:
Taking the best configuration (soft 16 uniform) and put it into different gamma (0.1, 1.0) then maybe testing with some extreme case (0.01 and 10)

As Gamma -> 0, the min reaches hard min and will be less blurry.
Gamma > 0, the loss would have negative value, but that is fine as it is simply mathematical artifacts.

Note: Focusing on understanding sdtw is uttermost important, so asides from doing more experiements, reading and understand the theory of sdtw
- original dtw
- log exp trick
- DP recursion
- soft min temprerature (used in sdtw)
- gradient calculations and substitution
- consistency in mathematical model (Notations) !!IMPORTANT!!
- what happens with different parameters in sdtw
- maybe do some hand calculation of the forward and backward pass


FIX - CHAPTER 2 and CHAPTER 1
- Citing Weighted SDTW and CUTURI Paper for the formulas
- Consistent Abberivation
- Fix notation issues
- Finish Chapter 1