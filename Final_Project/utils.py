import keras_uncertainty.backend as K
from keras.layers import Dense, Layer

activations = K.activations
conv_utils = K.conv_utils

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sklearn.metrics as M
import pandas as pd

def accuracy_prob_models(predictions, ground_truth):
  '''
  NB be careful at how the ground_truth vector is passed; check if it is a
  one-hot encoded vector or a list of ints.
  In the first case, it is necessary to run also this through the argmax(1)
  so we can recover a list of ints from it.
  '''
  return (predictions.argmax(1) == ground_truth).mean()

def confidence_prob_models(predictions):
  '''
  In this case we consider predictions as the probability value of the
  assignment class.
  '''
  return predictions.max(1)

def confidence_binning(confidence_vector, n_bins=10):
  '''
  Returns the bin index associated to each datapoint and the bin composition
  '''
  bins = np.linspace(1/n_bins, 1, n_bins)
  return np.digitize(confidence_vector, bins), bins

def reliability_vector(predictions, ground_truth, n_bins=10):
  '''
  Given predictions and ground truth, calculates the confidence scores associated
  with the predictions, bins the confidence into n_bins equispaced in the [0,1]
  line, then compute the per-bin accuracy.
  Returns a 1-d array of n_bins elements containing the per-bin accuracy, and an
  array containing the cutoffs of each bin.
  '''
  confidence_scores = confidence_prob_models(predictions)

  bins_composition, bins_cutoffs = confidence_binning(confidence_scores, n_bins)

  mean_accuracy_per_bins = np.full((n_bins,), fill_value=np.nan)
  bin_counts = np.bincount(bins_composition)

  for i in range(n_bins):
    if i > bins_composition.max():
      break
    if bin_counts[i] > 0:
      group_accuracy = accuracy_prob_models(
          predictions[bins_composition==i],
          ground_truth[bins_composition==i]
        )
      mean_accuracy_per_bins[i] = group_accuracy

  return mean_accuracy_per_bins, bins_cutoffs

def reliability_plot(reliability_vector, bins_cutoffs, clear_nans=True):
  bins_delta = bins_cutoffs[1] - bins_cutoffs[0]
  x_axis = bins_cutoffs - bins_delta/2

  if clear_nans:
    x_axis = x_axis[~np.isnan(reliability_vector)]
    reliability_vector = reliability_vector[~np.isnan(reliability_vector)]

  fig, ax = plt.subplots()
  ax.scatter(
      x_axis,
      reliability_vector
  )
  ax.set_xlim((0,1))
  ax.set_ylim((0,1))
  line = mlines.Line2D([0, 1], [0, 1], color='red')
  transform = ax.transAxes
  line.set_transform(transform)
  ax.add_line(line)
  ax.set_xlabel("confidence")
  ax.set_ylabel("accuracy")
  plt.plot(x_axis, reliability_vector)
  plt.show()

def ood_detection_scores(confidence_id, confidence_ood,
                         min_thresh=0.0, max_thresh=1.0, num_steps=101,
                         comparison_fn=np.less):

  ood_label = np.concatenate([
      np.zeros_like(confidence_id),
      np.ones_like(confidence_ood)
  ])

  len_id = len(confidence_id)

  threshold = []
  accuracy = []
  accuracy_id = []
  accuracy_ood = []
  tprs = [] 
  f1_score = []
  # add here other metrics

  for thresh in np.linspace(min_thresh, max_thresh, num_steps):
    detection_scores_id = np.where(comparison_fn(confidence_id, thresh), 1, 0)
    detection_scores_ood = np.where(comparison_fn(confidence_ood, thresh), 1, 0)
    detection_scores = np.concatenate([detection_scores_id, detection_scores_ood])

    acc = M.accuracy_score(ood_label, detection_scores)
    acc_id = M.accuracy_score(ood_label[:len_id], detection_scores_id)
    acc_ood = M.accuracy_score(ood_label[len_id:], detection_scores_ood)
    f1 = M.f1_score(ood_label, detection_scores)
    tpr = M.recall_score(ood_label, detection_scores)
    # add here other metrics

    threshold.append(thresh)
    accuracy.append(acc)
    accuracy_id.append(acc_id)
    accuracy_ood.append(acc_ood)
    tprs.append(tpr)
    f1_score.append(f1)
    # add here other metrics

  ood_detection_scores = pd.DataFrame({
    "threshold": threshold,
    "accuracy": accuracy,
    "accuracy_id": accuracy_id,
    "accuracy_ood": accuracy_ood,
    "true_positive_rate": tprs,
    "f1_score": f1_score,
    # add here other metrics
  })

  return ood_detection_scores

def plot_roc(confidence_id, confidence_ood, ax):
  ood_label = np.concatenate([
        np.zeros_like(confidence_id),
        np.ones_like(confidence_ood)
    ])

  true_pos_rate, false_pos_rate, thresh = M.roc_curve(
      ood_label,
      np.concatenate([confidence_id, confidence_ood])
  )
  roc_auc = M.auc(false_pos_rate, true_pos_rate)
  optimal_threshold = str(round(thresh[np.argmax(np.abs(true_pos_rate-false_pos_rate))], 3)) 
  display = M.RocCurveDisplay(fpr=false_pos_rate, tpr=true_pos_rate, roc_auc=roc_auc)
  display.plot(ax)
  ax.text(0.5, 0.9, f"opt thres: {optimal_threshold}", transform=ax.transAxes)