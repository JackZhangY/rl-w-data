from plots.utils import collect_mulit_run_files, plot_single_line, run_plot
import matplotlib.pyplot as plt

LOG_DIR = '../all_logs/rl-w-data/results'


COLORS = ['salmon', 'lime', 'deepskyblue', 'red', 'green', 'blue']
LINESTYLES = ['-', '-', '-', '-', '-', '-']
LEGENDS = ['vdice-nt=1-ea=iq', 'vdice-nt=1-ea=vd', 'vdice-nt=10-ea=iq', 'vdice-nt=10-ea=vd']

def generate_algos_dict():

    algos_dict = {
        'ValueDICE': [
            'num_trajs=1_absorbing=1_norm_obs=1_expert_algo=iq_learn',
            'num_trajs=1_absorbing=1_norm_obs=1_expert_algo=valuedice',
            'num_trajs=10_absorbing=1_norm_obs=1_expert_algo=iq_learn',
            'num_trajs=10_absorbing=1_norm_obs=1_expert_algo=valuedice',
        ],
    }

    return algos_dict

if __name__ == '__main__':
    env = 'Hopper-v2'
    algos = generate_algos_dict()
    item = 'returns/eval'
    length = 250
    million = 0.5

    run_plot(LOG_DIR, env, algos, item, length, million, COLORS, LEGENDS, LINESTYLES,
             'return', 0.1, show_legend=True, save_pdf=False)

    plt.show()