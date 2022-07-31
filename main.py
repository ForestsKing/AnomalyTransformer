import argparse

from exp.exp import Exp
from utils.setseed import set_seed

if __name__ == '__main__':
    set_seed(42)

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--temperature', type=int, default=50)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--windows_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_dir', type=str, default='./MSL/')
    parser.add_argument('--model_dir', type=str, default='./checkpoint/')
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--adjust', type=bool, default=False)

    config = vars(parser.parse_args())

    exp = Exp(config)
    # exp.train()
    exp.test()
