from data_provider.data_loader import Dataset_Weight
from torch.utils.data import DataLoader


def data_provider(args, flag):
    Data = Dataset_Weight
    train_only = args.train_only

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size

    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size

    data_set = Data(
        root_path=args.root_path,
        image_root = args.image_root,
        data_path = args.data_path,
        feature_path = args.feature_path,
        flag=flag,
        size=[args.seq_len, args.pred_len],
        features=args.features,
        target=args.target,
        train_only=train_only,
        image = args.image,
        text = args.text,
        text_from_img = args.text_from_img,
        b = args.breakfast,
        l = args.lunch,
        s = args.supper
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)
    return data_set, data_loader

