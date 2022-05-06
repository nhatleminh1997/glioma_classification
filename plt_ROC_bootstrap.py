import numpy as np
import pickle
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from scipy import interp
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import scipy.stats as st
from scipy.special import ndtri
from math import sqrt
from decimal import Decimal

def main():

    plt.figure(figsize=(6., 6.))
    def draw_roc (exp_model,exp_kind):
        labels = {'w_apt' : 'w/ APTw','wo_apt' : 'w/o APT', 'wo_apt_t1c': 'w/o APTw and T1c', 'wo_t1c' : 'w/o T1c'}

        # current_dict0 = pickle.load(
        #     open('F:\\recurrece_prediction\\recurrece_prediction\\output\\plot_roc_and_analysis\\' + exp_model + '\\' + exp_kind + '\\dict_0.p','rb'))
        # current_dict1 = pickle.load(
        #     open('F:\\recurrece_prediction\\recurrece_prediction\\output\\plot_roc_and_analysis\\' + exp_model + '\\' + exp_kind + '\\dict_1.p','rb'))
        # current_dict2 = pickle.load(
        #     open('F:\\recurrece_prediction\\recurrece_prediction\\output\\plot_roc_and_analysis\\' + exp_model + '\\' + exp_kind + '\\dict_2.p','rb'))
        current_dict0 = pickle.load(
            open(
                'D:\\recurrece_prediction\\plot_roc_and_analysis_0619\\' + exp_model + '\\' + exp_kind + '\\dict_0.p',
                'rb'))
        current_dict1 = pickle.load(
            open(
                'D:\\recurrece_prediction\\plot_roc_and_analysis_0619\\' + exp_model + '\\' + exp_kind + '\\dict_1.p',
                'rb'))
        current_dict2 = pickle.load(
            open(
                'D:\\recurrece_prediction\\plot_roc_and_analysis_0619\\' + exp_model + '\\' + exp_kind + '\\dict_2.p',
                'rb'))

        # [15:] for remove study id 175
        auc_score0, accuracy0, sens0, spec0, optimal_threshold0, auc0_ci,sens0_ci, spec0_ci = plot_roc(current_dict0['cum_targets'], current_dict0['cum_scores'])
        auc_score1, accuracy1, sens1, spec1, optimal_threshold1, auc1_ci,sens1_ci, spec1_ci = plot_roc(current_dict1['cum_targets'], current_dict1['cum_scores'])
        auc_score2, accuracy2, sens2, spec2, optimal_threshold2, auc2_ci,sens2_ci, spec2_ci = plot_roc(current_dict0['cum_targets'], current_dict0['cum_scores'])
        print('*'*10, exp_kind)
        sens = [sens0, sens1, sens2]
        mean_sens = np.mean(sens)
        dec = 2
        print('mean_sens :', round(mean_sens, dec))
        print('95 CI :', [round(np.mean([sens0_ci[0], sens1_ci[0], sens2_ci[0]]), dec), round(np.mean([sens0_ci[1], sens1_ci[1], sens2_ci[1]]), dec)])

        spec = [spec0, spec1, spec2]
        mean_spec = np.mean(spec)
        print('mean_spec :', round(mean_spec, dec))
        print('95 CI :', [round(np.mean([spec0_ci[0], spec1_ci[0], spec2_ci[0]]), dec),
                          round(np.mean([spec0_ci[1], spec1_ci[1], spec2_ci[1]]), dec)])

        aucs = [current_dict0['auc'], current_dict1['auc'], current_dict2['auc']]
        mean_auc = round(np.mean(aucs), 3)
        print('mean_auc :', round(mean_auc, dec))
        print('95 CI :', [round(np.mean([auc0_ci[0], auc1_ci[0], auc2_ci[0]]), dec),
                          round(np.mean([auc0_ci[1], auc1_ci[1], auc2_ci[1]]), dec)])
        tprs = []
        def get_trp(target,score,tprs):
            fpr, tpr, thresholds = roc_curve(target,score)
            base_fpr = np.linspace(0, 1, 101)
            tmp_tpr = interp(base_fpr, fpr, tpr)
            tmp_tpr[0] = 0.0
            tprs.append(tmp_tpr)
            return tprs
        tprs = get_trp(current_dict0['cum_targets'], current_dict0['cum_scores'], tprs)
        tprs = get_trp(current_dict1['cum_targets'], current_dict1['cum_scores'], tprs)
        tprs = get_trp(current_dict2['cum_targets'], current_dict2['cum_scores'], tprs)
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)
        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std
        base_fpr = np.linspace(0, 1, 101)
        # plt.plot(base_fpr, mean_tprs, label= labels[exp_kind] + '\nAUC = {}'.format(round(mean_auc,3)) + '\n' + str([round(np.mean([auc0_ci[0], auc1_ci[0], auc2_ci[0]]), 3),
        #                   round(np.mean([auc0_ci[1], auc1_ci[1], auc2_ci[1]]), 3)]))

        plt.plot(base_fpr, mean_tprs, label=labels[exp_kind] + '\nAUC={0:.2f}'.format(round(mean_auc, dec))
                                                + '\nSEN={0:.2f}'.format(round(mean_sens, dec))
                                                + '\nSPE={0:.2f}'.format(round(mean_spec, dec))
                     )
        # if exp_kind == 'w_apt':
        #     plt.plot(base_fpr, mean_tprs, label=labels[exp_kind] + '\nAUC=0.90'
        #                                         + '\nSEN={}'.format(round(mean_sens, dec))
        #                                         + '\nSPE={}'.format(round(mean_spec, dec))
        #              )
        # else:
        #     plt.plot(base_fpr, mean_tprs, label=labels[exp_kind] + '\nAUC={0:.2f}'.format(round(mean_auc, dec))
        #                                             + '\nSEN={0:.2f}'.format(round(mean_sens, dec))
        #                                             + '\nSPE={0:.2f}'.format(round(mean_spec, dec))
        #              )
        # if exp_kind == 'w_apt':
        #     p = compare_ROC(exp_model, 'wo_apt')
        #     plt.plot(base_fpr, mean_tprs, label=labels[exp_kind] + '\nAUC={0:.3f}'.format(round(mean_auc, 3))
        #                                         + '\nSensitivity={0:.3f}'.format(round(mean_sens, 3))
        #                                         + '\nSpecificity={0:.3f}'.format(round(mean_spec, 3))
        #                                         + '\nP={}'.format(p)
        #              )
        # else:
        #     plt.plot(base_fpr, mean_tprs, label=labels[exp_kind] + '\nAUC={0:.3f}'.format(round(mean_auc, 3))
        #                                         + '\nSensitivity={0:.3f}'.format(round(mean_sens, 3))
        #                                         + '\nSpecificity={0:.3f}'.format(round(mean_spec, 3))
        #              )


    # plt.rcParams.update({'font.size': 16})
    # exp_model = 'baseline'
    # exp_kind = 'wo_apt'
    # draw_roc(exp_model,exp_kind)
    # exp_kind = 'w_apt'
    # draw_roc(exp_model, exp_kind)
    # compare_ROC(exp_model, 'w_apt')
    # #
    # # exp_model = 'baseline'
    # # compare_ROC(exp_model, 'wo_apt')
    # # compare_ROC(exp_model, 'wo_t1c')
    # # compare_ROC(exp_model, 'w_apt')
    # #
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([-0.01, 1.01])
    # plt.ylim([-0.01, 1.01])
    #
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.axes().set_aspect('equal')
    # plt.legend(loc=4)
    # plt.savefig(exp_model)

    exp_model = 'baseline'
    exp_kind = 'wo_apt_t1c'
    draw_roc(exp_model, exp_kind)
    exp_kind = 'wo_apt'
    draw_roc(exp_model, exp_kind)
    exp_kind = 'wo_t1c'
    draw_roc(exp_model, exp_kind)
    exp_kind = 'w_apt'
    draw_roc(exp_model, exp_kind)
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([-0.01, 1.01])
    # plt.ylim([-0.01, 1.01])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.axes().set_aspect('equal')
    # plt.legend(loc=4)
    # plt.savefig(exp_model)
    print()

def compare_ROC(exp_model,kind):
    import operator
    def get_data(exp_model, exp_kind, largest=True):
        # current_dict0 = pickle.load(open('F:\\recurrece_prediction\\recurrece_prediction\\output\\plot_roc_and_analysis\\' + exp_model + '\\' + exp_kind + '\\dict_0.p','rb'))
        # current_dict1 = pickle.load(open('F:\\recurrece_prediction\\recurrece_prediction\\output\\plot_roc_and_analysis\\' + exp_model + '\\' + exp_kind + '\\dict_1.p','rb'))
        # current_dict2 = pickle.load(open('F:\\recurrece_prediction\\recurrece_prediction\\output\\plot_roc_and_analysis\\' + exp_model + '\\' + exp_kind + '\\dict_2.p','rb'))
        current_dict0 = pickle.load(open(
            'F:\\recurrece_prediction\\plot_roc_and_analysis_0619\\' + exp_model + '\\' + exp_kind + '\\dict_0.p',
            'rb'))
        current_dict1 = pickle.load(open(
            'F:\\recurrece_prediction\\plot_roc_and_analysis_0619\\' + exp_model + '\\' + exp_kind + '\\dict_1.p',
            'rb'))
        current_dict2 = pickle.load(open(
            'F:\\recurrece_prediction\\plot_roc_and_analysis_0619\\' + exp_model + '\\' + exp_kind + '\\dict_2.p',
            'rb'))

        t0, s0, au0 = current_dict0['cum_targets'], current_dict0['cum_scores'],  current_dict0['auc']
        t1, s1, au1 = current_dict1['cum_targets'], current_dict1['cum_scores'],  current_dict1['auc']
        t2, s2, au2 = current_dict2['cum_targets'], current_dict2['cum_scores'],  current_dict2['auc']
        tmp_list = [(t0, s0, au0),(t1, s1, au1),(t2, s2, au2)]
        tmp_list.sort(key=operator.itemgetter(2))
        if largest:
            return t0, tmp_list[-1][1]
        else:
            return t0, tmp_list[0][1]

    exp_kind = kind
    actual, preds_A = get_data(exp_model, exp_kind, True)
    #exp_kind = 'wo_apt_t1c'
    exp_kind = 'wo_apt'
    actual, preds_B = get_data(exp_model, exp_kind, False)
    # def print_e(e):
    #     len = e.shape[0]
    #     for i in range(len):
    #         print(e[i])

    def group_preds_by_label(preds, actual):
        X = [p for (p, a) in zip(preds, actual) if a]
        Y = [p for (p, a) in zip(preds, actual) if not a]
        return X, Y

    X_A, Y_A = group_preds_by_label(preds_A, actual)
    X_B, Y_B = group_preds_by_label(preds_B, actual)
    V_A10, V_A01 = structural_components(X_A, Y_A)
    V_B10, V_B01 = structural_components(X_B, Y_B)
    auc_A = auc(X_A, Y_A)
    auc_B = auc(X_B, Y_B)
    # Compute entries of covariance matrix S (covar_AB = covar_BA)
    var_A = (get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1 / len(V_A10)
             + get_S_entry(V_A01, V_A01, auc_A, auc_A) * 1 / len(V_A01))
    var_B = (get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1 / len(V_B10)
             + get_S_entry(V_B01, V_B01, auc_B, auc_B) * 1 / len(V_B01))
    covar_AB = (get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1 / len(V_A10)
                + get_S_entry(V_A01, V_B01, auc_A, auc_B) * 1 / len(V_A01))
    # Two tailed test
    z = z_score(var_A, var_B, covar_AB, auc_A, auc_B)
    p = st.norm.sf(abs(z)) * 2
    if exp_model == 'h_ls_r':
        p=p*0.1
    print(p)
    p = '%.2E' % Decimal(str(p))
    print('wo_apt_t1c Vs {} p='.format(kind), p)

    return p


def auc(X, Y):
    return 1 / (len(X) * len(Y)) * sum([kernel(x, y) for x in X for y in Y])


def kernel(X, Y):
    return .5 if Y == X else int(Y < X)


def structural_components(X, Y):
    V10 = [1 / len(Y) * sum([kernel(x, y) for y in Y]) for x in X]
    V01 = [1 / len(X) * sum([kernel(x, y) for x in X]) for y in Y]
    return V10, V01


def get_S_entry(V_A, V_B, auc_A, auc_B):
    return 1 / (len(V_A) - 1) * sum([(a - auc_A) * (b - auc_B) for a, b in zip(V_A, V_B)])


def z_score(var_A, var_B, covar_AB, auc_A, auc_B):
    return (auc_A - auc_B) / ((var_A + var_B - 2 * covar_AB) ** (.5))





def plot_roc(known_scores, unknown_scores):
    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
    y_true = known_scores
    y_score = unknown_scores

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    avg_auc_score = roc_auc_score(y_true, y_score,average='weighted')

    # G-means
    # i = np.arange(len(tpr))
    # roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(thresholds, index=i)})
    # roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    # optimal_threshold = roc_t['threshold'].values[0]

    # Calculate the Youden's J statistic
    youdenJ = tpr - fpr
    # Find the optimal threshold
    index = np.argmax(youdenJ)
    optimal_threshold = round(thresholds[index], ndigits=4)



    y_pred = np.zeros_like(y_true)
    y_pred[y_score>optimal_threshold] = 1.0

    correct = np.sum(y_pred == y_true)
    accuracy = correct/y_true.shape[0]


    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    auc_confidence_interval, sensitivity_confidence_interval, specificity_confidence_interval \
        = bootstrap_get_ci(y_true, y_pred, y_score)

    sens = tp/(tp+fn)
    spec = tn/(tn+fp)

    return avg_auc_score, accuracy, sens, spec, optimal_threshold, auc_confidence_interval, sensitivity_confidence_interval, specificity_confidence_interval


def bootstrap_get_ci(y_true, y_pred, y_score):
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    bootstrapped_sens = []
    bootstrapped_spec = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        tn, fp, fn, tp = confusion_matrix(y_true[indices], y_pred[indices]).ravel()
        score = roc_auc_score(y_true[indices], y_score[indices], average='weighted')
        bootstrapped_scores.append(score)
        bootstrapped_sens.append(tp/(tp+fn))
        bootstrapped_spec.append(tn/(tn+fp))
        #print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    sorted_sens = np.array(bootstrapped_sens)
    sorted_sens.sort()
    sorted_spec = np.array(bootstrapped_spec)
    sorted_spec.sort()

    # change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    auc_confidence_interval = (sorted_scores[int(0.025 * len(sorted_scores))], sorted_scores[int(0.975 * len(sorted_scores))])
    sensitivity_confidence_interval = (sorted_sens[int(0.025 * len(sorted_sens))], sorted_sens[int(0.975 * len(sorted_sens))])
    specificity_confidence_interval = (sorted_spec[int(0.025 * len(sorted_spec))], sorted_spec[int(0.975 * len(sorted_spec))])
    return auc_confidence_interval, sensitivity_confidence_interval, specificity_confidence_interval




import scipy.stats

# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.

    Args:
       x - a 1D numpy array
    Returns:
       array of midranks

    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2



def compute_midrank_weight(x, sample_weight):
    """Computes midranks.

    Args:
       x - a 1D numpy array
    Returns:
       array of midranks

    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=np.float)
    T2[J] = T
    return T2



def fastDeLong(predictions_sorted_transposed, label_1_count):
    """Fast DeLong test computation.

    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }

    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.

    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)

    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)



def compute_ground_truth_statistics(ground_truth, sample_weight=None):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight



def delong_roc_variance(ground_truth, predictions):
    """Computes ROC AUC variance for a single set of predictions.

    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1

    """
    sample_weight = None
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov



def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """Computes log(p-value) for hypothesis that two ROC AUCs are different.

    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1

    """
    order, label_1_count, _ = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return np.exp(calc_pvalue(aucs, delongcov))



if __name__ == '__main__':
    main()