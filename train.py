import os
import time
import argparse
import platform

import torch
import torch.optim as optim

from models.swin_transformer import swin_small_patch4_window7_224
from utils.utils import *
from utils.loss import OnlineReciprocalSoftmaxLoss
from utils.mining_utils import HardestNegativeTripletSelector


def main():
    args = parser()

    if platform.system() == 'Windows':
        num_workers = 0 
    else:
        num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # 在 Linux 系统下设置为 CPU 核心数


    train_dataloder, test_dataloder, train_dataset, test_dataset = getloader(args.data_root, 
                                                            args.batch_size, num_workers, 
                                                            args.train_ratio_txt)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    model = swin_small_patch4_window7_224(num_classes=train_dataset.getnumclass()).to(device)

    if args.model_path == 'weights/swin_small_patch4_window7_224.pth':
        assert os.path.exists(args.model_path), f"weights file: '{args.model_path}' not exist."
        weights_dict = torch.load(args.model_path, map_location=device)["model"]

        for k in list(weights_dict.keys()):
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
                if "fc_embedding" in k:
                    del weights_dict[k]
                if "fc_softmax" in k:
                    del weights_dict[k]
                if "norm_" in k:
                    del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.model_path == 'weights/current_model_state.pth' or args.model_path == 'weights/best_model_state.pth':
        weights_init = torch.load(args.model_path, map_location=device)["model_state"]
        model.load_state_dict(weights_init)
    

    triplet_selector = HardestNegativeTripletSelector(margin=args.triplet_margin)
    loss_function = OnlineReciprocalSoftmaxLoss(triplet_selector, args.lambda_factor)

    optimizer = optim.SGD(model.parameters(), lr=args.learn_rate, 
                          weight_decay=args.weight_decay)

    save_name = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
    save_path = os.path.join(args.save_folder, save_name+'.pth').replace('\\', '/')

    os.makedirs(args.log_folder, exist_ok=True)
    logger = getLogger(os.path.join(args.log_folder, save_name+'.txt').replace('\\', '/'))

    logger.info(f"using {len(train_dataset)} images for training, "
                f"{len(test_dataset)} images fot testidation.")

    for k, v in args.__dict__.items():  # save args
        logger.info("{}: {}".format(k, v))

    best_acc = 0.0
    accuracy_curr = 0.0

    for epoch in range(1, args.epochs + 1):

        loss, triplet_loss, softmax_loss, ap_distances, an_distances = train(model, train_dataloder, train_dataset, 
                                           device, loss_function, optimizer)

        if epoch % args.eval_freq == 0 or args.epochs-epoch <= 100:
            checkpoint_path = saveCheckpoint(args.save_folder, epoch, model, optimizer, 'current')
            accuracy_curr = test(args, checkpoint_path)

        if(accuracy_curr>best_acc):
            best_acc=accuracy_curr

            saveCheckpoint(args.save_folder, epoch, model, optimizer, 'best')
        
        logger.info(f"epoch: {epoch}/{args.epochs} "
                    f"loss: {loss:.4f} triplet_loss: {triplet_loss:.4f} softmax_loss: {softmax_loss:.4f} ap_distances: {ap_distances:.3f} an_distances: {an_distances:.3f} "
                    f"test_acc: {accuracy_curr:.3f}% best_acc: {best_acc:.3f}%")

    


def parser():
    parser = argparse.ArgumentParser(description="swt-triplet")

    parser.add_argument("--data_root", type=str, default='datasets/Cattle_12')

    parser.add_argument("--train_ratio_txt", type=str, default='dataset_split.txt')

    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--model_path", type=str, default='weights/swin_small_patch4_window7_224.pth')

    parser.add_argument("--learn_rate", type=float, default=0.0005)

    parser.add_argument('--embedding_size', type=int, default=128)
 
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    
    parser.add_argument("--lambda_factor", type=float, default=1)

    parser.add_argument('--triplet_margin', type=float, default=0.5)

    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument('--eval_freq', type=int, default=1)

    parser.add_argument("--save_folder", type=str, default='weights')
    
    parser.add_argument("--log_folder", type=str, default='logs')

    parser.add_argument('--save_embeddings', type=bool, default=False,
						help="Should we save the embeddings to file")

    return parser.parse_args()

if __name__ == '__main__':
    main()

