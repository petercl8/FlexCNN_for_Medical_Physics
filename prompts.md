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

Stage 0: Simplify codebase (in particular, #run_supervisory.py) so that it makes use of more code contained in functions and is easier to understand.
Stage 1: Update code (#dataset_classes.py, #stitching_notebook.ipynb, #run_supervisory.py, etc.) to allow for the optional loading of attenuation maps.
Stage 2: Update code to allow for tuning/training of the atten_map->atten_sinogram network ("attenuation network"). Checkpoint code could remain as it currently stands because we are just dealing with one network at this point.
Stage 3: Update code to allow for loading two networks (attenuation and activity) from checkpoints
Stage 4: Update code to allow for simultaneous training (of activity network), inference only (of attenuation network), and feature transfer. Would need to also include introduction of learnable scaling parameters for injected features.

This is only my plan and it may not be the right way to do things. I've never done something like this in Pytorch before, so I will need to dialogue with you about how we will accomplish this before we formalize our plan. Please repeat back to me your understanding of this task. Once I'm satisfied you understand the assignment, we can start talking about the plan.