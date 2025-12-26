## Note: This function still needs to be updated for SSIM and testing with the test set. See 'START HERE' comment below.
from FlexCNN_for_Medical_Physics.classes.generators import Generator
from FlexCNN_for_Medical_Physics.classes.discriminators import Disc_I_90, Disc_S_90
from FlexCNN_for_Medical_Physics.functions.helper.weights_init import weights_init_he

def run_CYCLE(config, checkpoint_dirPath=None, load_state=False, save_state=False):
    '''
    Note: Arguments are set to None/False to ensure that when RayTune calles train(), states are not saved/loaded. This uses up way too much hard drive space.
    Note: you may want to use 'model.train()' to put model back into training mode if you put it into eval mode at some point...
    '''

    ## Grab Stuff from Config Dict. ##
    # Convert batch_base2_exponent to batch_size for tune/train modes
    # Note: run_CYCLE doesn't receive settings dict, so we check if exponent key exists
    if 'batch_base2_exponent' in config:
        config['batch_size'] = 2 ** config['batch_base2_exponent']
    
    batch_size = config['batch_size']
    gen_b1 = config['gen_b1']
    gen_b2 = config['gen_b2']
    gen_lr = config['gen_lr']
    train_SI = config['train_SI']
    scale=config['SI_fixedScale'] if train_SI==True else config['SI_fixedScale']

    ## Tensorboard ##
    writer=SummaryWriter(tensorboard_dir)

    ## Initialize Generators/Discriminator/Summary Writer ##
    disc_I = Disc_I_90(config=config).to(device)
    disc_S = Disc_S_90(config=config).to(device)
    gen_SI = Generator(config=config, gen_SI=True).to(device)
    gen_IS = Generator(config=config, gen_SI=False).to(device)

    gen_both_opt = torch.optim.Adam(list(gen_SI.parameters()) + list(gen_IS.parameters()), lr=gen_lr, betas=(gen_b1, gen_b2)) # Common optimizer
    disc_I_opt = torch.optim.Adam(disc_I.parameters(), lr=config['SI_disc_lr'], betas=(config['SI_disc_b1'], config['SI_disc_b2']))
    disc_S_opt = torch.optim.Adam(disc_S.parameters(), lr=config['IS_disc_lr'], betas=(config['IS_disc_b1'], config['IS_disc_b2']))

    ## Load Data ##
    dataloader = DataLoader(
        NpArrayDataSet(image_path=image_path, sino_path=sino_path, config=config,
                       augment=augment, offset=offset, num_examples=num_examples, sample_division=sample_division,
                       recon1_path=recon1_path, recon2_path=recon2_path),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
    )

    ## Load Checkpoint ##
    if checkpoint_dirPath and load_state:
        # Load dictionary
        checkpoint = torch.load(os.path.join(checkpoint_dirPath, checkpoint_file))
        # Load values from dictionary
        start_epoch = checkpoint['epoch'] #If interrupted, this epoch may be trained more than once
        end_epoch = start_epoch + num_epochs
        batch_step = checkpoint['batch_step']
        gen_SI.load_state_dict(checkpoint['gen_SI_state_dict'])
        gen_IS.load_state_dict(checkpoint['gen_IS_state_dict'])
        gen_both_opt.load_state_dict(checkpoint['gen_both_opt_state_dict'])
        disc_I.load_state_dict(checkpoint['disc_I_state_dict'])
        disc_S.load_state_dict(checkpoint['disc_S_state_dict'])
        disc_I_opt.load_state_dict(checkpoint['disc_I_opt_state_dict'])
        disc_S_opt.load_state_dict(checkpoint['disc_S_opt_state_dict'])
        if run_mode=='test':
            gen_SI.eval()
            gen_IS.eval()

    ## START HERE WITH UPDATING THIS FUNCION FOR SSIM AND TEST SET FUNCTIONALITY

    else:
        print('Starting from scratch')
        start_epoch=0
        end_epoch=num_epochs
        batch_step = 0
        gen_SI = gen_SI.apply(weights_init_he)
        gen_IS = gen_IS.apply(weights_init_he)
        disc_I = disc_I.apply(weights_init_he)
        disc_S = disc_S.apply(weights_init_he)

    ## Loop Over Epochs ##
    for epoch in range(start_epoch, end_epoch):

        # Following variables reset every display step. The line below only establishes these variables, it does not reset them.
        mean_disc_loss, mean_adv_loss, mean_sup_loss, mean_cycle_loss, mean_pix_metric, mean_range_metric, mean_avg_metric = 0,0,0,0,0,0,0

        ## Loop Over Batches ##

        time_init_full = time.time()
        #time_init_loader = time.time()

        for sino_scaled, act_map_scaled, *recon_data in iter(dataloader): # Dataloader returns the batches. Loop over batches within epochs.

            #print(f'iter dataloader (time): {(time.time()-time_init_loader)*1000}')
            #print(f'FULL step (time): {(time.time()-time_init_full)*1000}')
            time_init_full = time.time()

            real_S = sino_scaled
            real_I = act_map_scaled

            ## Update Networks ##

            # Update Discriminators #
            # Image Discriminator #
            disc_I_opt.zero_grad() # Zero out the gradient before backpropagation
            with torch.no_grad(): # We won't be optmizing the generator here, so disabling gradients saves on resources
                fake_I = gen_SI(real_S)

            disc_I_loss = get_disc_loss(fake_I, real_I, disc_I, config['SI_disc_adv_criterion'])
            disc_I_loss.backward(retain_graph=True) # Update gradients
            disc_I_opt.step() # Update optimizer

            # Sinogram Discriminator #
            disc_S_opt.zero_grad() # Zero out the gradient before backpropagation
            with torch.no_grad(): # We won't be optmizing the generator here, so disabling gradients saves on resources
                fake_S = gen_IS(real_I)
            disc_S_loss = get_disc_loss(fake_S, real_S, disc_S, config['IS_disc_adv_criterion'])
            disc_S_loss.backward(retain_graph=True) # Update gradients
            disc_S_opt.step() # Update optimizer

            # Generators #
            gen_both_opt.zero_grad()
            gen_loss, adv_loss, sup_loss, cycle_loss, cycle_I, cycle_S = get_gen_loss(real_I, real_S, gen_IS, gen_SI, disc_I, disc_S, config)
            gen_loss.backward() # Update gradients
            gen_both_opt.step() # Update optimizer

            #print(f'update generator (time)): {(time.time()-time_init_gen)*1000}')

            ## Metrics ##
            # Pixel Distance #
            pix_metric_I = pixel_metric(real_I, fake_I)
            pix_metric_S = pixel_metric(real_S, fake_S)
            p_metric = pix_metric_I + pix_metric_S

            # Range Metric #
            range_metric_I = range_metric(real_I, fake_I)
            range_metric_S = range_metric(real_S, fake_S)
            r_metric = range_metric_I+range_metric_S

            # Average Metric #
            avg_metric_I = avg_metric(real_I, fake_I)
            avg_metric_S = avg_metric(real_S, fake_S)
            a_metric = avg_metric_I + avg_metric_S

            ## Running Statistics ##
            # Mean loss terms #
            mean_disc_loss    += (abs(disc_I_loss.item()) + abs(disc_S_loss.item())) / display_step
            mean_adv_loss     += abs(adv_loss) / display_step
            mean_sup_loss     += abs(sup_loss) / display_step
            mean_cycle_loss   += abs(cycle_loss) / display_step
            mean_pix_metric   += p_metric / display_step
            mean_range_metric += r_metric / display_step
            mean_avg_metric   += a_metric / display_step

            ## visualization CODE ##
            if batch_step % display_step == 1 and batch_step > 0: # runs if batch_step is a multiple of the display step

                # Optim_Metric #
                MS_Error = MSE(real_I, fake_I)
                loss_balance=abs(mean_adv_loss-mean_disc_loss)
                #optim_metric = 0.5*loss_balance+mean_cycle_loss+mean_pix_metric #+mean_avg_metric #+mean_range_metric
                optim_metric = MS_Error

                # Prune #
                #gen_SI = prune_gen(gen_SI)
                #gen_IS = prune_gen(gen_IS)

                ## Report  to Ray Tune ##
                if run_mode=='tune':
                    tune.report(batch_step=batch_step, epoch=epoch,
                                mean_adv_loss=mean_adv_loss, mean_disc_loss=mean_disc_loss, loss_balance=loss_balance,
                                mean_sup_loss=mean_sup_loss,
                                mean_cycle_loss=mean_cycle_loss,
                                mean_pix_metric=mean_pix_metric,
                                mean_avg_metric=mean_avg_metric,
                                optim_metric=optim_metric
                                )
                ## Display Stats & Images ##
                else:
                    print(f'================================================================================\nEPOCH: {epoch}, STEP: {batch_step}, Batch Size: {batch_size}')

                    lambda_adv, lambda_sup, lambda_cycle = config['lambda_adv'], config['lambda_sup'], config['lambda_cycle']

                    print(f'MSE (Images):  {MS_Error}')
                    print(f'lambda * Mean Adversarial Loss: {lambda_adv*mean_adv_loss}')
                    print(f'lambda * Mean Supervisory Loss: {lambda_sup*mean_sup_loss}')
                    print(f'lambda * Mean Cycle Loss      : {lambda_cycle*mean_cycle_loss}')
                    print(f'mean_disc_loss: {mean_disc_loss} // mean_adv_loss: {mean_adv_loss} // loss_balance (M) {loss_balance}')
                    print(f'mean_pix_metric (M): {mean_pix_metric}')
                    print(f'range_metric (M): {mean_range_metric}')
                    print(f'avg_metric: {mean_avg_metric}')
                    print(f'optim_metric: {optim_metric}')

                    ## visualize Images ##
                    # Images #
                    print('Ground Truth Images:')
                    show_single_unmatched_tensor(real_I)
                    print('Generated PET Images:')
                    show_single_unmatched_tensor(fake_I)
                    print('Cycle PET Images:')
                    show_single_unmatched_tensor(cycle_I)

                    # Sinograms #
                    print('Grount Truth Sinograms:')
                    show_single_unmatched_tensor(real_S) # low_rez_S = real
                    print('Generated Sinograms:')
                    show_single_unmatched_tensor(fake_S)
                    print('Cycle Sinograms:')
                    show_single_unmatched_tensor(cycle_S)

                    # Less interesting #
                    '''
                    print('Resized Model Images:')
                    show_single_unmatched_tensor(resized_I[0:9])
                    print('FBP, Full-Rez Sinograms, resized (90x90):')
                    show_single_unmatched_tensor(FBP_I[0:9])

                    print('Hi-Rez Sinograms:')
                    show_single_unmatched_tensor(high_rez_S)
                    print('Sinogram of Ground Truth Images:')
                    show_single_unmatched_tensor(project(ground_I))
                    print('Sinogram of Generated PET:')
                    show_single_unmatched_tensor(project(fake_I))
                    print('FBP, Low-Rez Sinograms:')
                    show_single_unmatched_tensor(reconstruct(low_rez_S, config['sino_size'], config['IS_normalize'], config['SI_fixedScale']))
                    '''

                    writer.add_scalar('mean adversarial loss', mean_adv_loss, batch_step)
                    writer.add_scalar('discriminator loss', mean_disc_loss, batch_step)
                    writer.add_scalar('loss balance', loss_balance, batch_step)
                    writer.add_scalar('pixel distance loss', mean_pix_metric, batch_step)
                    writer.add_scalar('cycle loss', mean_cycle_loss)
                    writer.add_scalar('supervisory loss (ground)', mean_sup_loss)
                    writer.flush()

                # Save State #
                if checkpoint_dirPath and save_state:
                    path = os.path.join(checkpoint_dirPath, checkpoint_file)
                    torch.save({
                        'epoch': epoch,
                        'batch_step': batch_step,
                        'gen_SI_state_dict': gen_SI.state_dict(),
                        'gen_IS_state_dict': gen_IS.state_dict(),
                        'gen_both_opt_state_dict': gen_both_opt.state_dict(),
                        'disc_I_state_dict': disc_I.state_dict(),
                        'disc_S_state_dict': disc_S.state_dict(),
                        'disc_I_opt_state_dict': disc_I_opt.state_dict(),
                        'disc_S_opt_state_dict': disc_S_opt.state_dict(),
                        }, path)

                # Zero Stats #
                mean_adv_loss = 0  # Should balance with mean_disc_loss (below)
                mean_disc_loss = 0 # Should balance with mean_adv_loss (above)
                mean_sup_loss_model = 0
                mean_sup_loss_ground = 0 #
                mean_cycle_loss = 0 # Better performing models will minimize this
                mean_pix_metric = 0 # Reasonable to minimize this for tuning purposes
                mean_range_metric=0
                mean_avg_metric=0

            batch_step += 1 #updates with every batch


            time_init_loader=time.time()
#call model.eval() before test set
