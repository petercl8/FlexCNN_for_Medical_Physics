Prompt 1:

I have a couple of big changes I need to make to #codebase that I'd like to discuss.  I have some serious brain fog today, so I'm hoping you can help me through this piece by piece. The two major changes are: 1) When tuning networks, I currently evaluate a network's performance by looking at its performance on the training set. I would like to change this so that it's performance is evaluated on a separate cross validation set. 2) Currently, reconstructions of sinograms are performed on the fly (for testing only in #sym:run_SUP ). These reconstructions are filtered back-projection (FBP) and maximum likelihood expectation maximization (MLEM) and are performed using functions I wrote. However, my new dataset not only includes a cross-val set, it also includes iterative reconstructions. Therefore, I can now load recons from the dataset. I don't want to delete any functions, but I would like the option to use iterative recons in place of the recons calculated on the fly.

First off, please repeat back to me what I would like to do so that I know you've go this. Then, which of these changes do you think we should make first? I have my own ideas, but I'd like to hear your logic.

Prompt 2:

Yes, that makes sense. Let's discuss how the data is structured. Currently, I have multiple numpy files. Right now I have one numpy file for sinograms and one for ground truth images. You can see in #file:dataset.py that both files are loaded and matched pairs are then output by the dataloader. My reconstructions are a natural extension of this. They are also matched. In other words (conceptually), sinogram[i] is matched with ground_truth[i], FORE_recon[i] and oblique_recon[i]. Here, grouind_truth[i] (which is currently called the "image") is the ground truth activity map, FOR_recon[i] is a Fourier rebinned reconstruction and oblique_recon[i] is a full 3D OSEM reconstruction. Does this make sense? Please confirm.

Prompt 3:

Great question. Even though the dataloader currently returns the unscaled sinogram and images, I don't believe these are currently used in the code. If you could varify this, I would appreciate it. If they are not, then we could simplify things and only return four total tensors. The activity map and reconstructions are all the same size but I would like to the code to take into account situations where they aren't. They can all be scaled to be the same size in the dataloader.

Prompt 4:
Yes, exactly. And let's also change the nomenclature. I'm thinking something like act_map_scaled. Some functions will need to be updated

Prompt 5:
That sounds good. We'll also need to update the metrics code. Could you look through this code (#file:metrics_wrappers.py , maybe #metrics.py) and propose changes?

Prompt 6:
I like the new function signature and logic flow inside the function. However, let's use generic names like Recon1/Recon2. Let's change the return signature accordingly (generic names) and update run_SUP(). Recon paths can go in the paths dict.