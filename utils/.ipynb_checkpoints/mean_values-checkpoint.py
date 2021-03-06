# Source: https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/mean.py

def get_mean(norm_value=255, dataset='activitynet'):

    # Below values are in RGB order
    assert dataset in ['activitynet', 'kinetics', 'ucf101', 'MRI','MRI_sub']

    if dataset == 'activitynet':
        return [114.7748/norm_value, 107.7354/norm_value, 99.4750/norm_value]
    elif dataset == 'kinetics':
        # Kinetics (10 videos for each class)
        return [110.63666788/norm_value, 103.16065604/norm_value, 96.29023126/norm_value]
    elif dataset == 'ucf101':
        return [101.00131/norm_value, 97.3644226/norm_value, 89.42114168/norm_value]

def get_std(norm_value=255):
    # Kinetics (10 videos for each class)
    return [38.7568578/norm_value, 37.88248729/norm_value, 40.02898126/norm_value]