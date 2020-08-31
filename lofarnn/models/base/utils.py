import argparse

def default_argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="whether to augment input data, default False",
    )
    parser.add_argument(
        "--norm",
        action="store_true",
        help="whether to normalize magnitudes or not, default False",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="whether to use single source or multiple, default False",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="whether to use shuffle multisource order, no effect for single source, default False",
    )
    parser.add_argument(
        "--num-sources", type=int, default=40, help="max number of sources to include",
    )
    parser.add_argument(
        "--nodes", type=int, default=1, help="max number of nodes to use",
    )
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument(
        "--num-trials", type=int, default=100, help="max number of trials to perform",
    )
    parser.add_argument(
        "--classes", type=int, default=40, help="max number of sources to include",
    )
    parser.add_argument(
        "--dataset", type=str, default="", help="path to dataset annotations files"
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="cross-entropy",
        help="loss to use, from 'cross-entropy' (default), 'focal', 'f1' ",
    )
    parser.add_argument("--experiment", type=str, default="", help="experiment name")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="number of minibatches between logging",
    )

    return parser