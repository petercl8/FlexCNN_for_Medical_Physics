import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from matplotlib.colors import Normalize
import os

def configure_plotting(plot_mode='inline'):
    """
    Configure matplotlib backend and environment based on plot_mode.
    
    Parameters:
    -----------
    plot_mode : str
        'never': Silent mode (Agg backend)
        'always': Interactive backend (TkAgg if not in Jupyter, else default)
        'inline': Only display in Jupyter/Interactive Window (default)
    """
    # Store pattern in environment for smart_show() to access
    os.environ['FLEXCNN_PLOT_MODE'] = plot_mode
    
    if plot_mode == 'never':
        # Never display plots
        matplotlib.use('Agg')
        print(f"[INFO] Plot mode: never - silent plotting (Agg backend)")
    elif plot_mode == 'always':
        # Always display plots
        try:
            get_ipython()
            print(f"[INFO] Plot mode: always - using interactive backend: {matplotlib.get_backend()}")
        except:
            matplotlib.use('TkAgg')
            print(f"[INFO] Plot mode: always - using TkAgg backend for window display")
    else:  # plot_mode == 'inline'
        # Only display in Jupyter/Interactive Window
        try:
            get_ipython()
            print(f"[INFO] Plot mode: inline - using interactive backend: {matplotlib.get_backend()}")
        except:
            matplotlib.use('Agg')
            print(f"[INFO] Plot mode: inline - terminal mode, silent plotting (Agg backend)")

# Smart plt.show() - displays inline in Jupyter, silent in regular terminal
def smart_show():
    """
    Display plots based on FLEXCNN_PLOT_MODE environment variable:
    - 'always': always show plots
    - 'inline': only show in Jupyter/Interactive Window
    - 'never': never show (silent)
    """
    plot_mode = os.environ.get('FLEXCNN_PLOT_MODE', 'inline')
    backend = plt.get_backend()
    
    if plot_mode == 'never':
        # Silent mode - never show
        return
    elif plot_mode == 'always':
        # Always show - use appropriate method for backend
        if 'ipykernel' in backend or 'inline' in backend:
            plt.show()
        else:
            plt.ion()
            plt.show(block=False)
            plt.pause(0.01)
    else:  # plot_mode == 'inline'
        # Only show in Jupyter/interactive backends
        if 'ipykernel' in backend or 'inline' in backend:
            plt.show()
        # else: silent (do nothing)

def show_single_unmatched_tensor(image_tensor, grid=False, cmap='inferno', fig_size=10):
    '''
    Function for visualizing images. The images are displayed, each with their own colormap scaling, so quantitative comparisons are not possible.
    Send only the images you want plotted to this function. Works with both single channel and multi-channel images.
    If using the single-channel grid option, it plots 120 images in a 15x8 grid.

    image_tensor:   image tensor of shape [num, chan, height, width]
    grid:           If True, displays images in a 15x8 grid (120 images in total). If false, images are displayed in a horizontal line.
    cmap:           Matplotlib color map
    fig_size:       figure size
    '''
    print(f'Shape: {image_tensor.shape} // Min: {torch.min(image_tensor)} // Max: {torch.max(image_tensor)} \
    //Mean Sum (per image): {torch.sum(image_tensor).item()/(image_tensor.shape[0]*image_tensor.shape[1])} // Sum (a single image): {torch.sum(image_tensor[0,0,:])}')

    #image_tensor = image_tensor.detach().squeeze(.cpu()
    image_tensor = image_tensor.detach().cpu()
    image_tensor = torch.clamp(image_tensor, min=0)

    num = image_tensor.size(dim=0)
    chan = image_tensor.size(dim=1)

    ## Plot 3-Channel Images ##
    #image_np = image_grid.mean(dim=0).squeeze().numpy() # This also works!

    ## Plot Multi-Channel Images ##
    if chan!=1:
        #123
        print(f'Mean (Ch 0): {torch.mean(image_tensor[:,0,:,:])} // Mean (Ch 1): {torch.mean(image_tensor[:,1,:,:])} // Mean (Ch 2): {torch.mean(image_tensor[:,2,:,:])}')

        # Plot Grid #
        if grid:
            fig, ax = plt.subplots(num, chan, figsize=(fig_size*num, fig_size*chan), constrained_layout=True)
            for N in range(0, num): # Iterate over image number
                for C in range(0, chan): # Iterate over channels
                    img = image_tensor[N,C,:,:]
                    ax[N,C].axis('off')
                    ax[N,C].imshow(img.squeeze(), cmap=cmap)

        # Plot in-Line #
        else:

            fig, ax = plt.subplots(1, num*(chan+1), figsize=(fig_size, fig_size*num*(chan+1)), constrained_layout=True)
            i=0
            for N in range(0, num): # Iterate over image number
                for C in range(0, chan): # Iterate over channels
                    img = image_tensor[N,C,:,:]
                    ax[i].axis('off')
                    ax[i].imshow(img.squeeze(), cmap=cmap)
                    i+=1
                blank = torch.ones_like(img)
                ax[i].axis('off')
                ax[i].imshow(blank.squeeze())
                i+=1

    ## Plot 1 Channel Images ##
    else:
        # Plot Grid #
        # Note: This plots 120 images at a time!
        if grid:
            cols, rows = 15, 8
        # Plot in-Line #
        else:
            rows = 1
            cols = image_tensor.shape[0]

        figure=plt.figure(figsize=(cols*fig_size,rows*fig_size))

        for i in range(0, cols*rows):
            img = image_tensor[i]             # Shape: torch.Size([3, 1, 180, 180]) /
            figure.add_subplot(rows,cols,i+1) # MatplotLib indeces start at 1
            plt.axis("off")
            plt.imshow(img.squeeze(), cmap=cmap)

    smart_show()

def show_multiple_unmatched_tensors(*image_tensors, cmap='inferno', fig_size=3):
    """
    Like show_multiple_matched_tensors, but each image gets its own normalization.
    No matching of min/max across tensors.
    """

    # Print stats for each tensor
    for tensor in image_tensors:
        print(f'Shape: {tensor.shape} // Min: {torch.min(tensor)} // Max: {torch.max(tensor)} '
              f'// Mean: {torch.mean(tensor)} // Mean Sum (per image): {torch.sum(tensor).item()/(tensor.shape[0]*tensor.shape[1])} '
              f'// Sum (a single image): {torch.sum(tensor[0,0,:])}')

    combined_tensor = torch.cat(image_tensors, dim=0).detach().cpu()
    combined_tensor = torch.clamp(combined_tensor, min=0)

    num_rows = len(image_tensors)
    num_cols = len(image_tensors[0])
    num_chan = image_tensors[0].size(dim=1)

    # -----------------------------
    # 1-CHANNEL IMAGES
    # -----------------------------
    if num_chan == 1:
        fig, ax = plt.subplots(num_rows, num_cols, squeeze=False,
                               figsize=(fig_size*num_cols, fig_size*num_rows),
                               constrained_layout=True)

        for row in range(num_rows):
            for col in range(num_cols):
                img = combined_tensor[row*num_cols + col, 0, :, :]
                img_min = torch.min(img).item()
                img_max = torch.max(img).item()
                norm = Normalize(vmin=img_min, vmax=img_max)

                ax[row, col].axis('off')
                ax[row, col].imshow(img.squeeze(), cmap=cmap, norm=norm)

    # -----------------------------
    # MULTI-CHANNEL IMAGES
    # -----------------------------
    else:
        print(f'Mean (Ch 0): {torch.mean(combined_tensor[:,0,:,:])} // '
              f'Mean (Ch 1): {torch.mean(combined_tensor[:,1,:,:])} // '
              f'Mean (Ch 2): {torch.mean(combined_tensor[:,2,:,:])}')

        # Layout: For each image, each channel + a blank divider
        fig, ax = plt.subplots(num_rows, num_cols*(num_chan+1), squeeze=False,
                               figsize=(fig_size*num_cols*(num_chan+1), fig_size*num_rows),
                               constrained_layout=True)

        i = 0
        for col in range(num_cols):    
            for chan in range(num_chan):
                for row in range(num_rows):
                    img = combined_tensor[row*num_cols + col, chan, :, :]
                    img_min = torch.min(img).item()
                    img_max = torch.max(img).item()
                    norm = Normalize(vmin=img_min, vmax=img_max)

                    ax[row, i].axis('off')
                    ax[row, i].imshow(img.squeeze(), cmap=cmap, norm=norm)
                i += 1

            # Divider before next multichannel image
            for row in range(num_rows):
                ax[row, i].axis('off')
                ax[row, i].imshow(torch.ones_like(img), cmap='gray')
            i += 1

    smart_show()

def show_multiple_matched_tensors(*image_tensors, cmap='inferno', fig_size=1.5):
    '''
    Function for visualizing images from multiple tensors. Each image is "matched" with images from the other tensors,
    and each matched set of images (one from each tensor) is plotted with the same colormap in a column.
    Send only the images you want plotted to this function. Works with both single channel and multi-channel images.

    image_tensors:  list of tensors, each of which may contain multiple images.
    '''
    for tensor in image_tensors:
        # Begin by printing statistics for each tensor
        print(f'Shape: {tensor.shape} // Min: {torch.min(tensor)} // Max: {torch.max(tensor)} \
        // Mean: {torch.mean(tensor)} // Mean Sum (per image): {torch.sum(tensor).item()/(tensor.shape[0]*tensor.shape[1])} // Sum (a single image): {torch.sum(tensor[0,0,:])}')

    combined_tensor = torch.cat(image_tensors, dim=0).detach().cpu()
    combined_tensor = torch.clamp(combined_tensor, min=0)

    num_rows = len(image_tensors)           # The number of rows equals the number of tensors (images to match)
    num_cols = len(image_tensors[0])        # The length of the zeroth element (of the list) is the number of images in a tensor.
    num_chan = image_tensors[0].size(dim=1) # Equivalent to: image_tensors[0].shape(1)

    ## Plot 1 Channel Images ##
    if num_chan==1:
        fig, ax = plt.subplots(num_rows, num_cols, squeeze=False, figsize=(fig_size*num_cols, fig_size*num_rows), constrained_layout=True)
        #fig, ax = plt.subplots(num_rows, num_cols, constrained_layout=True)

        i=0 # i = column number
        for col in range(0, num_cols): # Iterate over column number. All images in a column will have the same colormap.
            img_list=[]
            min_list=[]
            max_list=[]

            # Construct image list and normalization object for matched images in a column (iterating over rows) #
            for row in range(0,num_rows):                               # We iterate over rows in orcer
                img = combined_tensor[row*num_cols+col, 0 ,:,:]         # Grab the correct image (zeroth channel for 1-D images)
                img_list.append(img)                                    # We construct a new image list for each row
                min_list.append(torch.min(img).item())                  # Create list of image minimums
                max_list.append(torch.max(img).item())                  # Create list of image maximums
            norm = Normalize(vmin=min(min_list), vmax=max(max_list))    # We construct a normalization object with min/max = min/max pixel value for all images in list

            # Plot normalized images in a single column (iterating over rows) #
            for row in range(0,num_rows):
                ax[row, i].axis('off')
                ax[row, i].imshow(img_list[row].squeeze(), cmap=cmap, norm=norm) # Squeeze gets rid of extra channel dimension
            i+=1

    ## Plot Multi-Channel Images ##
    else:
        print(f'Mean (Ch 0): {torch.mean(combined_tensor[:,0,:,:])} // Mean (Ch 1): {torch.mean(combined_tensor[:,1,:,:])} // Mean (Ch 2): {torch.mean(combined_tensor[:,2,:,:])}')

        #if num_cols>3:  # Restricts to 3-channels. You could get rid of this without an issue.
        #    num_cols=3

        # Construct figure and axes. Note: 'num_chan+1' arises from the divider blank image btw. each multi-channel image
        fig, ax = plt.subplots(num_rows, num_cols*(num_chan+1), squeeze=False, figsize=(fig_size*num_cols*(num_chan+1), fig_size*num_rows), constrained_layout=True)

        i=0
        for col in range(0, num_cols):      # Iterate over column number
            for chan in range(0, num_chan): # Iterate over channels
                img_list=[]
                min_list=[]
                max_list=[]

                # Iterates over rows (one row per tensor) to construct an image list and normalization object a single column. All matched images have the same channel. #
                for row in range(0,num_rows):
                    img = combined_tensor[row*num_cols+col, chan ,:,:] # Constructs an image list where each row has the same channel #
                    img_list.append(img)
                    min_list.append(torch.min(img).item())
                    max_list.append(torch.max(img).item())
                norm = Normalize(vmin=min(min_list), vmax=max(max_list))

                # Iterates over rows to plot matched images in a single column. These share the same channel. #
                for row in range(0,num_rows):
                    ax[row, i].axis('off')
                    ax[row, i].imshow(img_list[row].squeeze(), cmap=cmap, norm=norm) # Squeeze gets rid of extra channel dimension
                i+=1

            # After all channels have been iterated, the complete multi-channel image has been plotted. Now we plot a divider before the next image #
            for row in range(0,num_rows):
                blank = torch.ones_like(img)
                ax[row, i].axis('off')
                ax[row, i].imshow(blank.squeeze())
            i+=1

    smart_show()

def show_single_commonmap_tensor(image_tensor, nrow=15, figsize=(27,18), cmap='inferno'):
    '''
    Function for visualizing images from one tensor, all of which will be plotted with the same scaled colormap. Only works with single-channel image tensors.

    image_tensor:  image tensor. nrow should go into this evenly.
    nrow:          number of rows for the image grid
    figsize:       figure size
    cmap:          color map
    '''
    tensor = torch.clamp(image_tensor, min=0).detach().cpu()
    image_grid = make_grid(tensor, nrow=nrow)  # from torchvision.utils import make_grid

    #print(f'Shape: {tensor.shape} // Min: {torch.min(tensor)} // Max: {torch.max(tensor)} \
    #// Mean: {torch.mean(tensor)} // Mean Sum (per image): {torch.sum(tensor).item()/(tensor.shape[0]*tensor.shape[1])} // Sum (a single image): {torch.sum(tensor[0,0,:])}')

    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.axis('off')

    image_grid = image_grid[0,:].squeeze()
    #plt.imshow(image_grid, cmap=cmap)
    im = ax.imshow(image_grid, cmap=cmap)
    #fig.colorbar(im, ax=ax)
    smart_show()

def show_multiple_commonmap_tensors(*image_tensors, cmap='inferno'):
    '''
    Function for visualizing images from multiple tensors, all of which will be plotted with the same scaled colormap. Only works with single-channel image tensors.

    *image_tensors: list of image tensors, all of which should contain the same number of images. Only send the number of images you want to plot to this function.
    '''
    # Print tensor statistics #
    for tensor in image_tensors:
        print(f'Shape: {tensor.shape} // Min: {torch.min(tensor)} // Max: {torch.max(tensor)} \
        // Mean: {torch.mean(tensor)} // Mean Sum (per image): {torch.sum(tensor).item()/(tensor.shape[0]*tensor.shape[1])} // Sum (a single image): {torch.sum(tensor[0,0,:])}')

    num_rows = len(image_tensors)
    num_columns = len(image_tensors[0])
    # Combine tensors into one & clamp #
    combined_tensor = torch.cat(image_tensors, dim=0).detach().cpu()
    combined_tensor = torch.clamp(combined_tensor, min=0)
    # Make a grid of the tensors #
    image_grid = make_grid(combined_tensor, nrow=num_columns) # Note: nrow is the number of images displayed in each row (i.e., the number of columns)

    # Determine figure size #
    print('num_rows:', num_rows)
    fig, ax = plt.subplots(1,1, figsize=(30,1*num_rows))
    #fig, ax = plt.subplots(1,1, figsize=(30,7))

    ax.axis('off')

    image_grid = image_grid[0,:].squeeze()
    im = ax.imshow(image_grid, cmap=cmap)
    fig.colorbar(im, ax=ax)
    smart_show()