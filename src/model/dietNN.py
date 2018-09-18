from argparse import ArgumentParser


def build_parser():
    par = ArgumentParser()
    par.add_argument('--m', type=str,
                     dest='model_load_path', help='filepath to load trained model', required=True)
    par.add_argument('--checkpoint_path', type=str,
                     dest='c', help='filepath to load checkpoints', required=True)
    return par


if __name__ == "__main__":
    parser = build_parser()
    options = parser.parse_args()
    model_load_path = options.model_load_path
    checkpoint_path = options.checkpoint_path

    print (model_load_path)
    print (checkpoint_path)