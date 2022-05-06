import numpy as np
import pickle
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from scipy import interp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import ndtri
from math import sqrt

def main():

    plt.figure(figsize=(5, 5))
    def draw_roc (exp_model,exp_kind):
        labels = {'w_apt' : 'w/ APTw','wo_apt' : 'w/o APT', 'wo_apt_t1c': 'w/o APTw and T1c', 'wo_t1c' : 'w/o T1c'}
        # current_dict0 = pickle.load(
        #     open('/mass/pengfei/recurrece_prediction/output/plot_roc_and_analysis/'+exp_model+'/'+exp_kind+'/dict_0.p', 'rb'))
        # current_dict1 = pickle.load(
        #     open('/mass/pengfei/recurrece_prediction/output/plot_roc_and_analysis/'+exp_model+'/'+exp_kind+'/dict_1.p', 'rb'))
        # current_dict2 = pickle.load(
        #     open('/mass/pengfei/recurrece_prediction/output/plot_roc_and_analysis/'+exp_model+'/'+exp_kind+'/dict_2.p', 'rb'))
        current_dict0 = pickle.load(
            open('F:\\recurrece_prediction\\recurrece_prediction\\output\\plot_roc_and_analysis\\' + exp_model + '\\' + exp_kind + '\\dict_0.p','rb'))
        current_dict1 = pickle.load(
            open('F:\\recurrece_prediction\\recurrece_prediction\\output\\plot_roc_and_analysis\\' + exp_model + '\\' + exp_kind + '\\dict_1.p','rb'))
        current_dict2 = pickle.load(
            open('F:\\recurrece_prediction\\recurrece_prediction\\output\\plot_roc_and_analysis\\' + exp_model + '\\' + exp_kind + '\\dict_2.p','rb'))

        auc_score0, accuracy0, sens0, spec0, optimal_threshold0, sens0_ci, spec0_ci = plot_roc(current_dict0['cum_targets'], current_dict0['cum_scores'])
        auc_score1, accuracy1, sens1, spec1, optimal_threshold1, sens1_ci, spec1_ci = plot_roc(current_dict1['cum_targets'], current_dict1['cum_scores'])
        auc_score2, accuracy2, sens2, spec2, optimal_threshold2, sens2_ci, spec2_ci = plot_roc(current_dict0['cum_targets'], current_dict0['cum_scores'])
        print('*'*10, exp_kind)
        sens = [sens0, sens1, sens2]
        mean_sens = np.mean(sens)
        print('mean_sens :', round(mean_sens, 3))
        print('95 CI :', [round(np.mean([sens0_ci[0], sens1_ci[0], sens1_ci[0]]), 3), round(np.mean([sens0_ci[1], sens1_ci[1], sens1_ci[1]]), 3)])
        # std_sens = np.std(sens)
        # print('std_sens :', round(std_sens, 3))
        # l = 1.960 * (std_sens / np.sqrt(3))
        # print('95 CI :', [round(mean_sens - l, 3), round(mean_sens + l, 3)])

        spec = [spec0, spec1, spec2]
        mean_spec = np.mean(spec)
        std_spec = np.std(spec)
        print('mean_spec :', round(mean_spec, 3))
        print('95 CI :', [round(np.mean([spec0_ci[0], spec1_ci[0], spec2_ci[0]]), 3),
                          round(np.mean([spec0_ci[1], spec1_ci[1], spec2_ci[1]]), 3)])
        # print('std_spec :', round(std_spec, 3))
        # l = 1.960 * (std_spec / np.sqrt(3))
        # print('95 CI :', [round(mean_spec - l, 3), round(mean_spec + l, 3)])

        aucs = [current_dict0['auc'], current_dict1['auc'], current_dict2['auc']]
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        print('mean_auc :', round(mean_auc,3))
        print('std_auc :', round(std_auc,3))
        l = 1.960 * (std_auc/np.sqrt(3))
        print('95 CI :', [round(mean_auc-l,3), round(mean_auc+l,3)])
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
        plt.plot(base_fpr, mean_tprs, label= labels[exp_kind] + '\nAUC = {}'.format(round(mean_auc,3)) + '\n' + str([round(mean_auc-l,3), round(mean_auc+l,3)]))
        # plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    # exp_model = 'h_ls'
    # exp_kind = 'wo_apt'
    # draw_roc(exp_model,exp_kind)
    # exp_kind = 'w_apt'
    # draw_roc(exp_model, exp_kind)
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([-0.01, 1.01])
    # plt.ylim([-0.01, 1.01])
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
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.axes().set_aspect('equal')
    plt.legend(loc=4)
    plt.savefig(exp_model)
    print()

def plot_roc(known_scores, unknown_scores):
    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
    y_true = known_scores
    y_score = unknown_scores

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    avg_auc_score = roc_auc_score(y_true, y_score,average='weighted')

    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(thresholds, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    #print('AUC {:.03f}'.format(auc_score))
    optimal_threshold = roc_t['threshold'].values[0]

    y_pred = np.zeros_like(y_true)
    y_pred[y_score>optimal_threshold] = 1.0

    correct = np.sum(y_pred == y_true)
    accuracy = correct/y_true.shape[0]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity_point_estimate, specificity_point_estimate, \
    sensitivity_confidence_interval, specificity_confidence_interval \
        = sensitivity_and_specificity_with_confidence_intervals(tp, fp, fn, tn)

    sens = tp/(tp+fn)
    spec = tn/(tn+fp)

    return avg_auc_score, accuracy, sens, spec, optimal_threshold, sensitivity_confidence_interval, specificity_confidence_interval




def _proportion_confidence_interval(r, n, z):
    """Compute confidence interval for a proportion.

    Follows notation described on pages 46--47 of [1].

    References
    ----------
    [1] R. G. Newcombe and D. G. Altman, Proportions and their differences, in Statisics
    with Confidence: Confidence intervals and statisctical guidelines, 2nd Ed., D. G. Altman,
    D. Machin, T. N. Bryant and M. J. Gardner (Eds.), pp. 45-57, BMJ Books, 2000.
    """

    A = 2 * r + z ** 2
    B = z * sqrt(z ** 2 + 4 * r * (1 - r / n))
    C = 2 * (n + z ** 2)
    return ((A - B) / C, (A + B) / C)


def sensitivity_and_specificity_with_confidence_intervals(TP, FP, FN, TN, alpha=0.95):
    """Compute confidence intervals for sensitivity and specificity using Wilson's method.

    This method does not rely on a normal approximation and results in accurate
    confidence intervals even for small sample sizes.

    Parameters
    ----------
    TP : int
        Number of true positives
    FP : int
        Number of false positives
    FN : int
        Number of false negatives
    TN : int
        Number of true negatives
    alpha : float, optional
        Desired confidence. Defaults to 0.95, which yields a 95% confidence interval.

    Returns
    -------
    sensitivity_point_estimate : float
        Numerical estimate of the test sensitivity
    specificity_point_estimate : float
        Numerical estimate of the test specificity
    sensitivity_confidence_interval : Tuple (float, float)
        Lower and upper bounds on the alpha confidence interval for sensitivity
    specificity_confidence_interval
        Lower and upper bounds on the alpha confidence interval for specificity

    References
    ----------
    [1] R. G. Newcombe and D. G. Altman, Proportions and their differences, in Statisics
    with Confidence: Confidence intervals and statisctical guidelines, 2nd Ed., D. G. Altman,
    D. Machin, T. N. Bryant and M. J. Gardner (Eds.), pp. 45-57, BMJ Books, 2000.
    [2] E. B. Wilson, Probable inference, the law of succession, and statistical inference,
    J Am Stat Assoc 22:209-12, 1927.
    """

    #
    z = -ndtri((1.0 - alpha) / 2)

    # Compute sensitivity using method described in [1]
    sensitivity_point_estimate = TP / (TP + FN)
    sensitivity_confidence_interval = _proportion_confidence_interval(TP, TP + FN, z)

    # Compute specificity using method described in [1]
    specificity_point_estimate = TN / (TN + FP)
    specificity_confidence_interval = _proportion_confidence_interval(TN, TN + FP, z)

    return sensitivity_point_estimate, specificity_point_estimate, sensitivity_confidence_interval, specificity_confidence_interval


if __name__ == '__main__':
    main()