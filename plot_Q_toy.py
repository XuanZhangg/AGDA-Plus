import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from ALG.Utils import *
import numpy as np


DATA_LIMIT = 500
PLOT_LIMIT = 500
stdx = 0
stdy = 0
b = 1
mu_y = 1
L = 20
for kappa in [int(L/mu_y)]:
    data_name = f'Q_stdx_{stdx}_stdy_{stdy}' + '_toy_muy_' + str(mu_y) + '_kappa_' + str(kappa) + f'_b_{b}'
    data_path = f'./result_data/{data_name}'
    for plot_part in ['z','lr_x','lr_y','ratio']:# ['x','y','z','loss','acc','lr_x','lr_y']:
        G = {}
        G['LS-GS-GDA-S'] = data_path +'/LS-GS-GDA-S'
        G['LS-GS-GDA'] = data_path +'/LS-GS-GDA'
        G['TiAda'] = data_path +'/TiAda'
        G['GS-GDA'] = data_path +'/AGDA'

        plt.figure(dpi=150)
        fig, ax = plt.subplots()
        is_log = False
        C = 0.0  # value center for log s

        for alg_name, file_name in G.items():
            data_xLimit = DATA_LIMIT
            plot_xLimit = PLOT_LIMIT
            with open(file_name, "rb") as fp:  # Unpickling
                record = pickle.load(fp)
                # load x-axis data
                oracle_complexity_counter = min(record['oracle_complexity_counter'], key=len)
                sample_complexity_counter = min(record['sample_complexity_counter'], key=len)
                iter_counter = min(record['iter_counter'], key=len)
                epoch_counter = min(record['epoch_counter'], key=len)
                total_oracle_complexity_counter = min(record['total_oracle_complexity_counter'], key=len)
                total_sample_complexity_counter = min(record['total_sample_complexity_counter'], key=len)
                total_iter_counter = min(record['total_iter_counter'], key=len)
                total_epoch_counter = min(record['total_epoch_counter'], key=len)
                #counter = total_oracle_complexity_counter[:data_xLimit]
                counter = total_iter_counter[:data_xLimit]
                data_xLimit = min(data_xLimit, len(counter))

                # load y-axis data
                valid_line_search = [i for i in range(len(record['acc'])) if len(record['acc'][i])>0]
                print(valid_line_search)

                acc = record['acc']
                acc = [acc[i][:data_xLimit] for i in valid_line_search]
                loss = [record['loss'][i][:data_xLimit] for i in valid_line_search]
                error = [[1 - ele[i] for i in range(len(acc[0]))] for ele in acc]
                lr_x = record['lr_x']
                lr_y = record['lr_y']
                lr_x = [lr_x[i][:data_xLimit] for i in valid_line_search]
                lr_y = [lr_y[i][:data_xLimit] for i in valid_line_search]
                lr_ratio  = [[ lr_y[i][j]/lr_x[i][j] for j in range(len(lr_x[i]))] for i in range(len(lr_x))]
                norm_sqaure_sto_grad_x = [record['norm_square_sto_grad_x'][i][:data_xLimit] for i in valid_line_search]
                norm_sqaure_sto_grad_y = [record['norm_square_sto_grad_y'][i][:data_xLimit] for i in valid_line_search]
                norm_sqaure_sto_grad_z = [[norm_sqaure_sto_grad_x[i][j] + norm_sqaure_sto_grad_y[i][j] for j in
                                           range(len(norm_sqaure_sto_grad_x[i]))] for i in
                                          range(len(norm_sqaure_sto_grad_x))]
                norm_sqaure_full_grad_x = [record['norm_square_full_grad_x'][i][:data_xLimit] for i in valid_line_search]
                norm_sqaure_full_grad_y = [record['norm_square_full_grad_x'][i][:data_xLimit] for i in valid_line_search]
                norm_sqaure_full_grad_z = [[norm_sqaure_full_grad_x[i][j] + norm_sqaure_full_grad_y[i][j] for j in
                                            range(len(norm_sqaure_full_grad_x[i]))] for i in
                                           range(len(norm_sqaure_full_grad_x))]

                norm_sqaure_sto_grad_x = normlize_data(norm_sqaure_sto_grad_x)
                norm_sqaure_sto_grad_y = normlize_data(norm_sqaure_sto_grad_y)
                norm_sqaure_sto_grad_z = normlize_data(norm_sqaure_sto_grad_z)
                norm_sqaure_full_grad_x = normlize_data(norm_sqaure_full_grad_x)
                norm_sqaure_full_grad_y = normlize_data(norm_sqaure_full_grad_y)
                norm_sqaure_full_grad_z = normlize_data(norm_sqaure_full_grad_z)

                contraction_times = record['contraction_times']
                #b = record['config'][-1]['b']
                N = record['config'][-1]['N']

                if plot_part == 'x':
                    shadowplot(counter, norm_sqaure_full_grad_x, label_input=alg_name, alpha=0.5, center=C, is_log=is_log,
                               is_var=True, alg_name=alg_name)
                elif plot_part == 'y':
                    shadowplot(counter, norm_sqaure_full_grad_y, label_input=alg_name, alpha=0.5, center=C, is_log=is_log,
                               is_var=True, alg_name=alg_name)
                elif plot_part == 'z':
                    shadowplot(counter, norm_sqaure_full_grad_z, label_input=alg_name, alpha=0.5, center=C, is_log=is_log,
                               is_var=True, alg_name=alg_name)
                elif plot_part == 'acc':
                    shadowplot(counter, error, label_input=alg_name, alpha=0.5, center=C, is_log=is_log, is_var=True,
                               alg_name=alg_name)
                elif plot_part == 'loss':
                    shadowplot(counter, loss, label_input=alg_name, alpha=0.5, center=C, is_log=is_log, is_var=True,
                               alg_name=alg_name)
                elif plot_part == 'lr_x':
                    shadowplot(counter, lr_x, label_input=alg_name, alpha=0.5, center=C, is_log=is_log, is_var=True,
                               alg_name=alg_name, is_step = 'LS' in alg_name, is_speical= alg_name=='GS-GDA',plot_part=plot_part)
                elif plot_part == 'lr_y':
                    shadowplot(counter, lr_y, label_input=alg_name, alpha=0.5, center=C, is_log=is_log, is_var=True,
                               alg_name=alg_name, is_step = 'LS' in alg_name, is_speical= alg_name=='GS-GDA',plot_part=plot_part)
                elif plot_part == 'ratio':
                    shadowplot(counter, lr_ratio, label_input=alg_name, alpha=0.5, center=C, is_log=is_log, is_var=True,
                               alg_name=alg_name, is_step = 'LS' in alg_name, is_speical= alg_name=='GS-GDA',plot_part=plot_part)

                
        # if 'lr_x' in plot_part:
        #     plt.plot(range(plot_xLimit), [1/L/kappa**2]*plot_xLimit, label = '$1/(L\kappa^2)$', linestyle = '-.', color='red', linewidth=2)
        
        # if 'lr_y' in plot_part:
        #     plt.plot(range(plot_xLimit), [1/L]*plot_xLimit, label = '$1/L$', linestyle = '-.', color='red', linewidth=2)

        if 'ratio' in plot_part:
            plt.plot(range(plot_xLimit), [kappa]*plot_xLimit, label = '$\kappa$', linestyle = '-.', color='red', linewidth=2)


        if plot_part == 'x':
            plt.legend(fontsize=15, loc='upper right')
        elif plot_part == 'y':
            plt.legend(fontsize=15, loc='upper right')
        elif plot_part == 'z':
            plt.legend(fontsize=15, loc='lower left')
        elif plot_part == 'acc':
            plt.legend(fontsize=15, loc='upper right')
        elif plot_part == 'loss':
            plt.legend(fontsize=15, loc='upper right')
        elif plot_part == 'lr_x':
            plt.legend(fontsize=15, loc='lower right')
        elif plot_part == 'lr_y':
            plt.legend(fontsize=15, loc='center right')
        elif plot_part == 'ratio':
            plt.legend(fontsize=15, loc='lower right')

        plt.xlabel("Number of gradient calls", fontsize=15)

        if plot_part == 'x':
            plt.ylabel(r"$\frac{||\nabla_x\mathcal{L}(x_k,y_k)||^2}{||\nabla_x\mathcal{L}(x_0,y_0)||^2}$", fontsize=15)
        elif plot_part == 'y':
            plt.ylabel(r"$\frac{||\nabla_y\mathcal{L}(x_k,y_k)||^2}{||\nabla_y\mathcal{L}(x_0,y_0)||^2}$", fontsize=15)
        elif plot_part == 'z':
            plt.ylabel(r"$\|\nabla\mathcal{L}(x_k,y_k)||^2$", fontsize=15)
            #plt.ylabel(r"$\frac{||\nabla\mathcal{L}(x_k,y_k)||^2}{||\nabla\mathcal{L}(x_0,y_0)||^2}$", fontsize=15)
        elif plot_part == 'acc':
            plt.ylabel(r"Train Error", fontsize=15)
        elif plot_part == 'loss':
            plt.ylabel(r"Loss", fontsize=15)
        elif plot_part == 'lr_x':
            plt.ylabel(r"Primal step size: $\tau$", fontsize=15)
        elif plot_part == 'lr_y':
            plt.ylabel(r"Dual step size: $\sigma$", fontsize=15)
        elif plot_part == 'ratio':
            plt.ylabel(r"Ratio: $\sigma/\tau$", fontsize=15)

        # set label size here
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)

        # set x,y range here
        #
        # plt.ylim(1e-2,)
        plt.xlim(0,plot_xLimit)


        # set personalized axis scale here
        if plot_part == 'x':
            plt.yscale('log')
        elif plot_part == 'y':
            plt.yscale('log')
        elif plot_part == 'z':
            plt.ylim(1e-120,100)
            plt.yscale('log')
        elif plot_part == 'acc':
            plt.ylim(0, 0.6)
        elif plot_part == 'loss':
            plt.yscale('log')
        elif plot_part == 'lr_x':
            plt.yscale('log')
        elif plot_part == 'lr_y':
            plt.yscale('log')
        elif plot_part == 'ratio':
            plt.yscale('log')

        # plt.xscale('log')
        if is_log:
            ax.set_yticklabels([round(np.exp(y) + C, 2) for y in ax.get_yticks()], fontsize=10)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.xaxis.offsetText.set_visible(True)
        plt.grid()
        # set title here
        # plt.title('Qudradic_Bilinear_Obj',fontsize = 15)

        data_name_tmp = list(data_name)
        for i in range(len(data_name_tmp)):
            if data_name_tmp[i] == '.':
                data_name_tmp[i] = '_'

        plt.savefig(f'./figure/{"".join(data_name_tmp)}_{plot_part}.pdf', bbox_inches='tight', facecolor='w', dpi=150)
