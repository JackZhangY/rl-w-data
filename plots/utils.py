import os
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator



plt.style.use('seaborn-paper')

radius = 20

def collect_mulit_run_files(file_path, setting_prefix):
    file_list = []
    for folder in os.listdir(file_path):
        if folder.startswith(setting_prefix):
            file_list.append(os.path.join(file_path, folder))
    return  file_list

def plot_single_line(file_list, item,  color, label, linestyle, length, million):
    smooth_scores = []
    cur_value_data_list = []
    cur_length_list = []
    convkernel_1 = np.ones(2 * radius + 1)
    for log_dir in file_list:
        data_log = event_accumulator.EventAccumulator(log_dir)
        data_log.Reload()
        try:
            value_log = data_log.scalars.Items(item)
        except KeyError:
            return
        value_data = [i.value for i in value_log]
        cur_value_data_list.append(value_data)
        cur_length_list.append(len(value_data))
    cur_min_length = min(cur_length_list)
    print(cur_min_length)

    cur_min_length = min(cur_min_length, length)
    for v_data in cur_value_data_list:
        _smooth_v1_data = np.convolve(v_data, convkernel_1, mode='same') \
                          / np.convolve(np.ones_like(v_data), convkernel_1, mode='same')
        smooth_scores.append(_smooth_v1_data[:cur_min_length])


    x_data = np.arange(0, cur_min_length) / cur_min_length * (million * cur_min_length / length)
    ymean = np.mean(smooth_scores, axis=0)
    ystd = np.std(smooth_scores, axis=0)
    ystderr = ystd / np.sqrt(len(smooth_scores))
    plt.plot(x_data, ymean, color=color, linestyle=linestyle, label=label)
    plt.fill_between(x_data, ymean - ystderr / 1, ymean + ystderr / 1, color=color, alpha=.2)


def run_plot(results_dir, env, algos, item, length, million, COLORS, LEGENDS, LINESTYLES,
                y_label, x_interval=1, show_legend=False, save_pdf=False):
    """
    plot all the performance line, and set the plot properties
    """

    # plot all lines
    env_dir = os.path.join(results_dir, env)
    idx = 0
    plt.figure(figsize=(12, 9), dpi=80)
    for alg_name, alg_params in algos.items():
        for param in alg_params:
            files = collect_mulit_run_files(os.path.join(env_dir, alg_name), param)
            plot_single_line(files, item, COLORS[idx], LEGENDS[idx], LINESTYLES[idx], length, million)
            idx += 1

    # plot setting
    plt.xlabel('Timesteps($\\times 10^6$)', fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    if show_legend:
        plt.legend(loc=4, shadow=True, fontsize=20)
    plt.grid(ls='--')
    plt.rc('grid', linestyle='-.')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=20)
    plt.xlim((0, million))
    ax = plt.gca()
    x_major_locator = MultipleLocator(x_interval)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.title(env, fontsize=28)

    if save_pdf:
        log_property = item.split('/')[-1]
        plt.savefig('./pdfig/{}_{}.pdf'.format(env, log_property), bbox_inches='tight', figsize=(8, 6))


