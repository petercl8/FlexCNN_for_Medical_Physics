def ROI_NEMA_hot(ground_truth_tensor, reconstruction_tensor, background_mask, hot_mask):
   
    background_truth =          np.sum(np.multiply(background_mask.numpy(), ground_truth_tensor.numpy())) # You must convert tensors to numpy arrays before multiplication
    background_reconstruction = np.sum(np.multiply(background_mask.numpy(), reconstruction_tensor.numpy()))

    hot_truth          = np.sum(np.multiply(hot_mask.numpy(), ground_truth_tensor.numpy()))
    hot_reconstruction = np.sum(np.multiply(hot_mask.numpy(), reconstruction_tensor.numpy()))

    print('background_reconstruction: ', background_reconstruction)
    print('background_truth: ', background_truth)
    print('hot_reconstruction: ', hot_reconstruction)
    print('hot_truth: ', hot_truth)

    x = 100*(hot_reconstruction/background_reconstruction-1)/(hot_truth/background_truth-1) # This measure is invariant to absolute pixel values. It only measure relative contrast.

    return x


def ROI_NEMA_cold(reconstruction_tensor, background_mask, cold_mask):

    background_reconstruction = np.sum(np.multiply(background_mask.numpy(), reconstruction_tensor.numpy()))
    cold_reconstruction = np.sum(np.multiply(cold_mask.numpy(), reconstruction_tensor.numpy()))

    print('background_reconstruction: ', background_reconstruction)
    print('cold_reconstruction: ', cold_reconstruction)
    x = 100*(1-cold_reconstruction/background_reconstruction) # This measure does not take into account ground truths because the cold/background should ideally always be zero.

    return x