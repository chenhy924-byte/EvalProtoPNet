import os
import model
import torch
import argparse
from util.eval_interpretability import evaluate_consistency
import multiprocessing
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=str, default='0')
    parser.add_argument('--data_set', default='Barefoot_Dataset', type=str)
    parser.add_argument('--data_path', type=str, default='datasets/Barefoot_Dataset/')
    parser.add_argument('--nb_classes', type=int, default=-1)
    parser.add_argument('--test_batch_size', type=int, default=30)
    parser.add_argument('--num_prototypes_per_class', type=int, default=10)

    # Model
    parser.add_argument('--base_architecture', type=str, default='vgg16')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--prototype_shape', nargs='+', type=int, default=[2000, 64, 1, 1])
    parser.add_argument('--prototype_activation_function', type=str, default='log')
    parser.add_argument('--add_on_layers_type', type=str, default='regular')

    parser.add_argument('--resume', type=str)
    args = parser.parse_args()

    if args.gpuid:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
    img_size = args.input_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        # Align prototype count with checkpoint to avoid shape mismatch
        ckpt_num_proto = int(checkpoint['model']['prototype_vectors'].shape[0])
        args.prototype_shape[0] = ckpt_num_proto
        # Infer classes from checkpoint if possible
        if ckpt_num_proto % args.num_prototypes_per_class == 0:
            inferred_classes = ckpt_num_proto // args.num_prototypes_per_class
            if args.nb_classes <= 0:
                args.nb_classes = inferred_classes

    if args.nb_classes <= 0 and args.data_set == 'Barefoot_Dataset':
        train_dir = os.path.join(args.data_path, 'train_cropped_augmented')
        if not os.path.isdir(train_dir):
            train_dir = os.path.join(args.data_path, 'train_cropped')
        args.nb_classes = len([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        args.prototype_shape[0] = args.nb_classes * args.num_prototypes_per_class

    # Load the model
    ppnet = model.construct_OursNet(base_architecture=args.base_architecture,
                                  pretrained=True, img_size=img_size,
                                  prototype_shape=args.prototype_shape,
                                  num_classes=args.nb_classes,
                                  prototype_activation_function=args.prototype_activation_function,
                                  add_on_layers_type=args.add_on_layers_type)
    ppnet = ppnet.to(device)
    ppnet_multi = torch.nn.DataParallel(ppnet)

    if checkpoint is not None:
        ppnet.load_state_dict(checkpoint['model'])

    consistency_score = evaluate_consistency(ppnet, args)
    print('Consistency Score : {:.2f}%'.format(consistency_score))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()