I already have an existing PyTorch class called Generator that builds a configurable encoder–decoder CNN (U-Net–like) from a configuration dictionary called config. You may find that in #generators.py.
The Generator:

Builds encoder, bottleneck, and decoder stages entirely from config

Already supports skip connections, channel distributions, dropout, etc.

Accepts an input tensor of shape C x 288 x 288

Produces an output tensor of shape C x 180 x 180 (cropped from a final 288 x 288 layer)

Is already used to train a sinogram -> activity image reconstruction network

I want to reuse this exact Generator class with minimal code changes.

Goal:
Train an attenuation map -> attenuation sinogram network using the same Generator and training code. After training, freeze its weights and inject its intermediate feature maps into a second Generator that performs sinogram -> activity reconstruction.

Important constraints:

Do not change Generator internals if possible

Assume spatial feature map sizes always match; if they do not, that is an error

Minimize new code and architectural changes

Gradients must never flow into the attenuation network

Architecture plan:

After tuning/training the attenuation nettwork, we instantiate two Generator objects:

atten_gen: attenuation map -> attenuation sinogram (frozen), loaded from checkpoint.

recon_gen: sinogram -> activity image (tunable/trainable)

Both generators are built using their own config dictionaries

Feature injection:

Capture intermediate feature maps from atten_gen during its forward pass

Inject those features into recon_gen at three locations:

Bottleneck -> bottleneck (always included)

Attenuation decoder -> reconstruction encoder (mid-level, may be toggled on or off)

Attenuation encoder -> reconstruction decoder (mid-level, may be toggled on or off)

Feature injection is done by concatenation along the channel dimension

Scaling:

Each injected feature tensor must be multiplied by a trainable per-channel scaling factor

Scaling parameter shape should be (C, 1, 1)

Initialize scaling factors to a small value (e.g. 0.1)

Scaling parameters belong to the reconstruction network only

Implementation guidance:

Other than feature injection (and the attendent change in channels at those locations), the reconstruction Generator should work exactly as before (as if feature injection were disabled)

Minimal, well-commented code for lateral feature injection

No unnecessary refactoring or architectural changes

Objective:
This implements a frozen-backbone, teacher–student style architecture where structural features learned from attenuation maps softly guide the sinogram -> image reconstruction without co-adaptation.

For our plan:

Let's discuss the specifics of how this will be implemented. ChatGPT had some suggestions:

a) Use forward hooks or minimal wrapper logic to extract intermediate features

b) A lightweight wrapper or subclass that connects two Generator instances

My current (tentative thought) is to divide this task into a number of stages that we can complete one at a time:

Stage 0: Simplify #codebase (in particular, #run_supervisory.py) so that it makes use of more code contained in functions and is easier to understand.

Stage 1: Update code (#dataset_classes.py, #stitching_notebook.ipynb, #run_supervisory.py, etc.) to allow for the optional loading of attenuation maps.

Stage 2: If loading attenuation maps, we also need to "load" their sinogram pairs. I figure the best way to do this (to avoid storing unnecessary data) is to create the sinograms on the fly. The attenuation maps can be transformed in real time using a radon transform to create the attenuation sinograms. I already have some code written which does forward projection (in #reconstruction_projection). We need to update that code to create these sinograms that are of approximately the same dimensions as the activity sihnograms. We could then call that code from the dataloader to load sinograms and return them (along with everything else) to the trainable. That would keep code generation nicely siloed.

Stage 3: Update code to allow for tuning/training of the atten_map->atten_sinogram network ("attenuation network"). Checkpoint code could remain mostly as it currently stands. We already have a lot of search dictionaries in #search_spaces.py . I'd prefer not to complicate things unless necessary. The current scheme involved merging one or more dictionaries depending upon the tesk. This occurs in #file:construct_dictionaries.py.  For a supervisory SI tuning, config_RAY_SI is combined with config_RAY_SUP and then either config_RAY_SI_learnScale or config_RAY_IS_fixedScale. This scaffolding was originally written for cycleGANs but I don't see why we couldn't use the exact same plan for this new scheme. All hyperparameters which belong determine (attenuation map)->(attenuation sinogram) architecture can start with IS_* keys. Any hyperparameters related to the fitting model are contained within Config_RAY_SUP dictionary. These keys can be reused, first for training the attenuation network and then for the activity network. There should be no issue because once the attenuation network is frozen, it won't need hyperparameters like gen_lr, gen_b1, etc. Therefore, when checkpointing, you should only ever need one dictionary of hyperparameters for both networks. I also believe that this can all be done without breaking future uses of this scheme for CycleGANs. Let me know if this isn't clear or, even if it is, if you think another plan would be better.

Stage 4: Update code to allow for loading two networks (attenuation and activity) from checkpoints. Do we need to save network weights in separate files for the two networks? I'm not sure. Since we are training the networks sequentially (not simultaneously) I suspect we may need to add that complication but I'm not sure.

Stage 5: Update code to allow for simultaneous training (of activity network), inference only (of attenuation network), and feature transfer. Here, we also need to include introduction of learnable scaling parameters for injected features. There should also be options to turn off one or more of those feature injections for albation. I suppose you could think of the toggles as hyperparameters that could be tuned. We should create a new dictionary that applies only that apply in dual network use. Something like:

config_RAY_DUAL = {
    'inject_bottleneck': True,  # Always injected (not tunable)
    'inject_dec_to_enc': tune.choice([True, False]),  # Ablation toggle
    'inject_enc_to_dec': tune.choice([True, False]),  # Ablation toggle
    'feature_scale_init': 0.1,  # Initial value for learnable scaling params
    'feature_scale_lr_mult': tune.loguniform(0.1, 10.0),  # LR multiplier for scaling params
}

This is only my plan and it may not be the right way to do things. I've never done something like this in Pytorch before, so I will need to dialogue with you about how we will accomplish this before we formalize our plan. Please repeat back to me your understanding of this task. Once I'm satisfied you understand the assignment, we can start talking about the plan.


----

Further Considerations:

1) Only the smallest layer post-activation features in the neck need to be injected. If a neck has multiple layers of the same size, the last one before upsampling should be chosen.
2) I need to understand these options more. Are you proposing A) one dict/one checkpoint file B) two dicts/two checkpoint files?
3) We'd only be using one optimizer at a time. config_RAY_DUAL dictionary could be merged in #file:construct_dictionaries.py  with the other dicts to create a single search space.


----

A) You seem to have made the decision already to checkpoint with separate files. I still need to be convinced that's the best option. It's certainly easier for me to uncomment one dictionary than to keep track of two when switching between experiments in scientific code.
B) There is one complication we still need to deal with. One of the hyperparameters ('SI_gen_neck' or 'IS_gen_neck') controls neck size. For the attenuation-->sinogram network, we should pick the smallest neck. That way, no matter what the neck size of the sinogram->activity network, we will have features that can be injected at the appropriate sizes. We can discuss this more when it's time to implement.
C) We should do everything for the 288x288 network first.
D) Do the stages I propose make sense to you? If not, please propose new ones. I would like to go stage by stage in our plan. For each stage, please break up the stage into multiple steps. These steps should each allow for distinct code edits where the code can be tested after each step.

---

1) For workflow, I anticipate tuning/training the attenuation network once. Then, I can do many experiments with the activity portion. Therefore, it makes sense to split the saved weight values into two files. However, keeping all hyperparameters in a single dict simplifies things. Does this make sense to you?

2) Let's simplify this drastically. We can tune/train the attenuation network at medium neck size (5x5). Then we can simply inject outside the neck to the activity network (which can have any neck size). Here are the proposed connections: 9x9 attenuation encoder side -> 9x9 activity decoder side. 9x9 attenuation decoder side --> 9x9 activity encoder side. Same pattern but with 9x9 attenuation encoder side -> 9x9 activity decoder side. 9x9 attenuation decoder side --> 9x9 activity encoder side. And then same pattern but with 36x36 and 144x144 layers.

3) Agree, let's defer the others.