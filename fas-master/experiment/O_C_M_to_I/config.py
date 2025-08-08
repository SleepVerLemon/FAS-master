class DefaultConfigs(object):
    seed = 666
    # SGD
    weight_decay = 5e-4
    momentum = 0.9
    # learning rate
    init_lr = 0.0005
    lr_epoch_1 = 20
    lr_epoch_2 = 80

    # model
    pretrained = True
    model = 'resnet18'     
    # training parameters
    gpus = "1"
    batch_size = 10

    max_iter = 4000
    lambda_app = 0.5
    lambda_adreal = 0.01
    # test model name

    tgt_best_model_name1 = 'net_model_best_0.22_32.pth.tar' 
    tgt_best_model_name2 = 'net2_model_best_0.22_32.pth.tar' 
    
    # source data information
    src1_data = 'casia'
    src1_train_num_frames = 1
    src2_data = 'oulu'
    src2_train_num_frames = 1
    src3_data = 'msu'
    src3_train_num_frames = 1
    # target data information
    tgt_data = 'replay'
    tgt_test_num_frames = 2
    # paths information
    checkpoint_path = './' + tgt_data + '_checkpoint/' + model + '/DGFANet/'
    best_model_path = './' + tgt_data + '_checkpoint/' + model + '/best_model/'
    logs = './logs/'

config = DefaultConfigs()
