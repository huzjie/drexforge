# Calibration: confidence that matches accuracy

A decision model is only useful inside a larger system if its *confidence* is
trustworthy: when it says "0.9 sure", it should be right ~90% of the time.
Expected Calibration Error (ECE) measures exactly that.

## Formula

```
ECE = sum_over_bins  (n_bin / n) * | acc_bin - conf_bin |
```

Bin decisions by their confidence, compare the mean confidence in each bin to
the actual accuracy, weight by bin size. Lower is better.

## The reference's calibration story

At skill 0.5 the model is random but *overconfident* (sharp softmax over noise),
so `confidence ~0.69` while `accuracy ~0.25` -> ECE 0.440. As skill rises toward
1.0, accuracy and confidence converge, and ECE falls to 0.007.

## Reusable notes

1. **A decision temperature is what makes confidence meaningful.** Without it,
   softmax over logits in `[0,1]` gives the winner only ~0.47 even when it's
   right. Scale logits by a temperature `T` before softmax so a confident,
   correct model concentrates mass near 1.0.
2. **Calibrate against accuracy, not against "was it right once".** Use ECE
   (binned), not a single confidence number.
3. **A random-but-overconfident model is a real failure mode.** Track ECE so a
   "confident wrong" model shows up as high ECE rather than looking fine.

See `drexforge/bench.py` (`_ece`) and `examples/calibration_curve.py`.
