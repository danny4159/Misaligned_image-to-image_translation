def train(opt):
    import time
    from data import CreateDataLoader
    from models import create_model
    from util.visualizer import Visualizer
    import os

    model = create_model(opt)

    #Loading data
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('Training images = %d' % dataset_size)
    visualizer = Visualizer(opt) # Web Visualizer
    total_steps = 0

    #Starts training
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()
            #Save current images (real_A, real_B, fake_B)
            if  epoch_iter % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch,epoch_iter, save_result)
            #Save current errors   
            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)
            #Save model based on the number of iterations
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')
    
            iter_data_time = time.time()
        #Save model based on the number of epochs
        print(opt.dataset_mode)
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)
    
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
    if opt.dataset_misalign == True: # misalign을 적용했을 때만 MI(Mutual information)를 출력 (MI: misalign의 정도를 정량화)
        print("Average Mutual Information: ", data_loader.average_mi)
        mi_path = os.path.join(opt.checkpoints_dir, opt.name, 'Avg MI.txt')
        with open(mi_path, 'w') as f:
            f.write(f"Average Mutual Information: {data_loader.average_mi}\n")


def test(opt):
    import sys
    import os
    from data import CreateDataLoader
    from models import create_model
    from util.visualizer import Visualizer
    from util import html
    import numpy as np
    
    sys.argv.extend(['--serial_batches'])
    
    # opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle

    
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    psnr_list = []
    ssim_list = []
    # fid_list = []
    for i, data in enumerate(dataset): # data는 dictionary 형태로, data는 batch 1일때 input_A: [1,3,256,256] input_B: [1,1,256,256] 총 910 slices
        if i >= opt.how_many:
            break
        model.set_input(data)
        psnr, ssim = model.test() #TODO: FID_score는 추후 추가 (현재는 구현 실패함)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        # fid_list.append(fid_score)
        visuals = model.get_current_visuals() # 여기 들어가기 전에 npy 저장하고 psnr, ssim 구하도록
        img_path = model.get_image_paths()
        img_path[0]=img_path[0]+str(i)
        if i % 50 == 0: # 50의 배수만 진행상황 출력
            print('%04d: process image... %s' % (i, img_path))
        visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)    
    webpage.save()
    print("Average PSRN: ", np.mean(psnr_list))
    print("Average SSIM: ", np.mean(ssim_list))
    eval_path = os.path.join(opt.results_dir, opt.name, 'PSNR,SSIM,FID eval.txt')
    with open(eval_path, 'w') as f:
        f.write(f"Average PSNR: {np.mean(psnr_list)}\n")
        f.write(f"Average SSIM: {np.mean(ssim_list)}\n")
    # print(fid_score_list)
    # print(np.mean(fid_score_list))