import os
import model
import torch
import argparse
from util.eval_interpretability import evaluate_stability
import multiprocessing
import random
import numpy as np
    

def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Best-effort deterministic behavior (some ops may still be nondeterministic on GPU)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=str, default='0')
    parser.add_argument('--data_set', default='Barefoot_Dataset', type=str)
    parser.add_argument('--data_path', type=str, default='datasets/Barefoot_Dataset/')
    parser.add_argument('--nb_classes', type=int, default=-1)
    parser.add_argument('--test_batch_size', type=int, default=30)
    parser.add_argument('--num_prototypes_per_class', type=int, default=10)
    parser.add_argument('--half_size', type=int, default=36, help='half size of bbox used to map prototypes to parts (affects consistency/stability)')
    parser.add_argument('--noise_seed', type=int, default=0, help='base seed for random noise used in stability evaluation')
    parser.add_argument('--stability_trials', type=int, default=1, help='number of random-noise trials to average for stability score')

    # Model
    parser.add_argument('--base_architecture', type=str, default='resnet34')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--prototype_shape', nargs='+', type=int, default=[2000, 64, 1, 1])
    parser.add_argument('--prototype_activation_function', type=str, default='log')
    parser.add_argument('--add_on_layers_type', type=str, default='regular')

    parser.add_argument('--resume', type=str)
    args = parser.parse_args()

    set_determinism(int(args.noise_seed))

    if args.gpuid:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
    img_size = args.input_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        ckpt_num_proto = int(checkpoint['model']['prototype_vectors'].shape[0])
        args.prototype_shape[0] = ckpt_num_proto
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

    stability_score = evaluate_stability(ppnet, args, half_size=args.half_size)
    print('Stability Score : {:.2f}%'.format(stability_score))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()