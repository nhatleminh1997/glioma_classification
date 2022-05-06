import numpy as np
import pickle
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from scipy import interp
import matplotlib.pyplot as plt


def main():
    def analysis_spec_sen(exp_model,exp_kind):
        current_dict0 = pickle.load(
            open('/mass/pengfei/recurrece_prediction/output/plot_roc_and_analysis/' + exp_model + '/' + exp_kind + '/dict_0.p','rb'))
        current_dict1 = pickle.load(
            open('/mass/pengfei/recurrece_prediction/output/plot_roc_and_analysis/' + exp_model + '/' + exp_kind + '/dict_1.p','rb'))
        current_dict2 = pickle.load(
            open('/mass/pengfei/recurrece_prediction/output/plot_roc_and_analysis/' + exp_model + '/' + exp_kind + '/dict_2.p','rb'))
        aucs = [current_dict0['auc'], current_dict1['auc'], current_dict2['auc']]

        def analysis(y_true,y_score,):
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            specificity = 1 - fpr
            spec_ = np.zeros((1,3))

            sen_ = np.zeros((1, 3))
            # find coresponding sensitivity at diff specificity
            # 70%
            absolute_val_array = np.abs(specificity - 0.7)
            smallest_difference_index = absolute_val_array.argmin()
            spec = specificity[smallest_difference_index]
            sens = tpr[smallest_difference_index]
            spec_[0,0] = sens

            # 80%
            absolute_val_array = np.abs(specificity - 0.8)
            smallest_difference_index = absolute_val_array.argmin()
            spec = specificity[smallest_difference_index]
            sens = tpr[smallest_difference_index]
            spec_[0,1] = sens
            # 90%
            absolute_val_array = np.abs(specificity - 0.9)
            smallest_difference_index = absolute_val_array.argmin()
            spec = specificity[smallest_difference_index]
            sens = tpr[smallest_difference_index]
            spec_[0,2] = sens

            # find coresponding specificity at diff sensitivity
            # 70%
            absolute_val_array = np.abs(tpr - 0.7)
            smallest_difference_index = absolute_val_array.argmin()
            spec = specificity[smallest_difference_index]
            sens = tpr[smallest_difference_index]
            sen_[0,0] = spec
            # 80%
            absolute_val_array = np.abs(tpr - 0.8)
            smallest_difference_index = absolute_val_array.argmin()
            spec = specificity[smallest_difference_index]
            sens = tpr[smallest_difference_index]
            sen_[0,1] = spec
            # 90%
            absolute_val_array = np.abs(tpr - 0.9)
            smallest_difference_index = absolute_val_array.argmin()
            spec = specificity[smallest_difference_index]
            sens = tpr[smallest_difference_index]
            sen_[0,2] = spec
            return spec_, sen_

        spec_0, sen_0 = analysis(current_dict0['cum_targets'], current_dict0['cum_scores'])
        spec_1, sen_1 = analysis(current_dict1['cum_targets'], current_dict1['cum_scores'])
        spec_2, sen_2 = analysis(current_dict2['cum_targets'], current_dict2['cum_scores'])
        spec = np.concatenate((spec_0,spec_1,spec_2),0)
        sen = np.concatenate((sen_0,sen_1,sen_2),0)
        # find coresponding sensitivity at diff specificity
        print(exp_kind)
        print('specificity')
        mean_spec = np.mean(spec,0)
        print(np.round(mean_spec, 3))
        std_spec = np.std(spec, 0)
        l = 1.960 * (std_spec / np.sqrt(3))
        print('95 CI :', [np.round(mean_spec - l, 3), np.round(mean_spec + l, 3)])

        print('sensitivity')
        mean_spec = np.mean(sen, 0)
        print(np.round(mean_spec, 3))
        std_spec = np.std(sen, 0)
        l = 1.960 * (std_spec / np.sqrt(3))
        print('95 CI :', [np.round(mean_spec - l, 3), np.round(mean_spec + l, 3)])
        print()

    exp_model = 'h_ls'
    exp_kind = 'wo_apt'
    analysis_spec_sen(exp_model, exp_kind)
    exp_kind = 'w_apt'
    analysis_spec_sen(exp_model, exp_kind)


if __name__ == '__main__':
    main()