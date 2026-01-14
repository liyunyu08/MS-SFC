from common.utils import set_seed


def dataset_builder(args):
    set_seed(args.seed)  # fix random seed for reproducibility

    if args.dataset == 'miniimagenet':
        from models.dataloader.imagenet import ImageNet as Dataset
    elif args.dataset == 'cub':
        #from models.dataloader.cub200 import CUB as Dataset
        # from models.dataloader.cub import CUB as Dataset
        from models.dataloader.cub_box import DatasetLoader as Dataset
    elif args.dataset == 'tieredimagenet':
        from models.dataloader.tiered_imagenet import tieredImageNet as Dataset
    elif args.dataset == 'cifar_fs':
        from models.dataloader.cifar_fs import DatasetLoader as Dataset
    elif args.dataset == 'dogs':
        from models.dataloader.dogs import Dogs as Dataset
    elif args.dataset == 'cars':
        from models.dataloader.cars import Cars as Dataset
    elif args.dataset == 'aircraft':
        from models.dataloader.aircraft import Aircraft as Dataset
    elif args.dataset == 'airport':
        from models.dataloader.airport import Airport as Dataset
    elif args.dataset == 'flowers':
        from models.dataloader.flowers import Flowers as Dataset
    elif args.dataset == 'fc100':
        from models.dataloader.fc100 import FC100 as Dataset
        
    elif args.dataset == 'tiered_meta_iNat':
        from models.dataloader.fc100 import FC100 as Dataset
    else:
        raise ValueError('Unkown Dataset')
    return Dataset
