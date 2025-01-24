from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \
    Dataset_Pred, Dataset_fm, Dataset_ts, UTSDataset, TrickDataset
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
    'fm': Dataset_fm,
    'ts': Dataset_ts,
    'pd': Dataset_Pred,
    'utsd': UTSDataset,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size   # bsz=1 for evaluation
        #batch_size = 1   # bsz=1 for evaluation
        
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq
    if(args.fm and not args.zs):
            data_set = Data(subset_name=r'UTSD-1G', flag=flag, input_len=args.seq_len,output_len=args.pred_len)
    else:
        if args.use_ims:
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.output_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
            )
        else:           
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
            )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
