class DefaultConfigs(object):
    seed = 999
    # SGD
    weight_decay = 5e-4
    momentum = 0.9
    # learning rate
    init_lr = 0.00001
    lr_epoch_1 = 20
    lr_epoch_2 = 100
    lr_epoch_3 = 250
    # model
    pretrained = False
    model = 'resnet18'     # resnet18 or maddg
    # training parameters
    gpus = "0"
    # batch_size = 10
    norm_flag = True
    max_iter = 4000
    lambda_triplet = 1
    lambda_adreal = 0.5
    # test model name
    #tgt_best_model_name = 'model_best_0.08_29.pth.tar' 
    # tgt_best_model_name = 'model_best_0.16233_397.pth.tar'#'model_best_0.14826_24.pth.tar'
    tgt_best_model_name1 = 'net_model_best_0.17569_53.pth.tar' 
    tgt_best_model_name2 = 'net2_model_best_0.17569_53.pth.tar' 
    # 
    # source data information
    src1_data = 'casia'
    src1_train_num_frames = 1
    src2_data = 'replay'
    src2_train_num_frames = 1
    src3_data = 'msu'
    src3_train_num_frames = 1
    # target data information
    tgt_data = 'oulu'
    tgt_test_num_frames = 2
    # paths information
    checkpoint_path = '/root/FAS_model_xiaopang114/fas-master/experiment/I_C_M_to_O/' + tgt_data + '_checkpoint/' + model + '/DGFANet/'
    best_model_path = '/root/FAS_model_xiaopang114/fas-master/experiment/I_C_M_to_O/' + tgt_data + '_checkpoint/' + model + '/best_model/'
    logs = './logs/'
    
    data_path = "/root/classification/Datasets/CIM_O/imagenet/"
    input_size = 256
    batch_size = 64
    lr = 0.00005
    epoch = 100
    lambda_app = 0.1
    num_workers = 4
    pin_mem = True
    color_jitter = 0.4
    aa = "rand-m9-mstd0.5-inc1"  # 自动增强策略
    train_interpolation = "bicubic"  # 训练插值方式
    reprob = 0.25  # 随机擦除概率
    remode = "pixel"  # 随机擦除模式
    recount = 1  # 随机擦除次数
    distributed = False  # 是否分布式训练
    gpu = 0  # GPU编号
    dist_url = 'tcp://127.0.0.1:23456'
config = DefaultConfigs()
