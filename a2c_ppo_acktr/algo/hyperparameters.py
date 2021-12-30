import argparse


def specific_algo_params(common_parser):

    specific_parser = common_parser.add_subparsers()

    parserA = specific_parser.add_parser('ValueDICE', help='ValueDICE specific hyperparameters')
    parserA.add_argument('--il_algo', type=str, default='ValueDICE')
    parserA.add_argument('--nu_lr', type=float, default=1e-3, help='learning rate for nu(Q) network')
    parserA.add_argument('--actor_lr', type=float, default=1e-5, help='learning rate for actor network')
    parserA.add_argument('--replay_regularization', type=float, default=0.1, help='coefficient for replay regularization')
    parserA.add_argument('--nu_regularization', type=float, default=10.0, help='coefficient for gradient penalty when update nu network')
    parserA.add_argument('--updates_per_step', type=int, default=1, help='number of updates per env step')
    parserA.add_argument('--num_random_actions', type=int, default=2e3,
                             help='number of implementing random actions when filling the replay buffer')
    parserA.add_argument('--hidden_size', type=int, default=256, help='hidden size for MLP networks')

    # todo: add other algorithms hyperparameters

    return common_parser.parse_args()



