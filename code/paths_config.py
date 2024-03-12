dataset_paths = {
	#  Face Datasets (In the paper: FFHQ - train, CelebAHQ - test)
	#'ffhq': 'datasets/e4e_ffhq_encoder.pt',
	'ffhq': '/home/sh/pycharm_workspace/dataset/ffhq',
	'celeba_test': '/home/sh/pycharm_workspace/dataset/celeba_hq/val',

	'cifar10': '/home/sh/pycharm_workspace/stylegan2-ada-unlearn/dataset/cifar10',
	'cifar10_test': '/home/sh/pycharm_workspace/stylegan2-ada-unlearn/dataset/cifar10_test',

	#  Cars Dataset (In the paper: Stanford cars)
	'cars_train': 'datasets/e4e_cars_encoder.pt',
	'cars_test': '',

	#  Horse Dataset (In the paper: LSUN Horse)
	'horse_train': 'datasets/e4e_horse_encoder.pt',
	'horse_test': '',

	#  Church Dataset (In the paper: LSUN Church)
	'church_train': 'datasets/e4e_church_encoder.pt',
	'church_test': '',

	#  Cats Dataset (In the paper: LSUN Cat)
	'cats_train': '',
	'cats_test': ''
}

model_paths = {
	'stylegan_ffhq': '../pretrained_models/stylegan2-ffhq-config-f.pt',
	'ir_se50': '../pretrained_models/model_ir_se50.pth',
	'shape_predictor': '../pretrained_models/shape_predictor_68_face_landmarks.dat',
	'moco': '../pretrained_models/moco_v2_800ep_pretrain.pt',
	'stylegan_cifar10':'/home/sh/pycharm_workspace/stylegan2-ada-unlearn/pretrained_model/cifar/cifar10.pkl'
}
