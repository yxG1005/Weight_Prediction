from data_provider.data_loader import Dataset_Weight
from torch.utils.data import DataLoader

data_dict = {
    'weight':Dataset_Weight
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    train_only = args.train_only

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
        data_path=args.data_path
    # elif flag == 'pred':
    #     shuffle_flag = False
    #     drop_last = False
    #     batch_size = 1
    #     freq = args.freq
    #     data_path="test_weight.csv"
    #     Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
        data_path=args.data_path

    data_set = Data(
        root_path=args.root_path,
        image_root = args.image_root,
        data_path=data_path,
        feature_path = args.feature_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale = args.scale,
        timeenc=timeenc,
        freq=freq,
        train_only=train_only,
        image = args.image,
        text = args.text,
        text_from_img = args.text_from_img,
        b = args.breakfast,
        l = args.lunch,
        s = args.supper
    )
    # print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader

