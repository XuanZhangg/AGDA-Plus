import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from ALG.Utils import *
import numpy as np

Q_show_limit_x = {10:30000,20:30000,50:30000,100:30000,1000:30000,10000:30000}
sinQ_show_limit_x = {10:10000,20:10000,50:10000,100:20000,1000:30000,10000:30000}

DATA_LIMIT = 500000
PLOT_LIMIT = 30000
for is_last in [True,False]:
    stdx = 0 # 0.1 for stochastic Q, 0 for determintisc Q
    stdy = 0
    b = 1
    mu_y = 1
    for kappa in [10]:#,20,50,100,1000,10000]:# 5,10,20,50,100,1000,10000
        print(f"processing kappa={kappa}")
        data_name = f'Q_stdx_{stdx}_stdy_{stdy}' + '_muy_' + str(mu_y) + '_kappa_' + str(kappa) + f'_b_{b}'
        data_path = f'./result_data/{data_name}'
        if "sin" in data_name:
            PLOT_LIMIT = sinQ_show_limit_x[kappa]
        else:
            PLOT_LIMIT = Q_show_limit_x[kappa]

        for plot_part in ['z']:#,'lr_x','lr_y']:# ['x','y','z','loss','acc','lr_x','lr_y']:
            G = {}
            # G['GS-GDA-B,N=2'] = data_path +'/primal_line_search_N_2_AGDA'
            # G['GS-GDA-B,N=5'] = data_path +'/primal_line_search_N_5_AGDA'
            # #G['LS-GS-GDA-R'] = data_path +'/LS-GS-GDA-R'
            # # G['LS-GS-GDA-S-R'] = data_path + '/LS-GS-GDA-S-R'
            G['LS-GS-GDA-S'] = data_path + '/LS-GS-GDA-S'
            G['LS-GS-GDA'] = data_path +'/LS-GS-GDA'
            G['TiAda'] = data_path +'/TiAda'
            G['NeAda'] = data_path + '/NeAda'
            G['GS-GDA-B,N=1'] = data_path +'/primal_line_search_N_1_AGDA'
            G['PF_AGP_NSC'] = data_path +'/PF-AGP-NSC'
            G['J-GDA'] = data_path +'/GDA'
            G['GS-GDA'] = data_path +'/AGDA'
            G['Smooth-AGDA'] = data_path + '/Smooth-AGDA'

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
                    if 'GS-GDA-B' in alg_name:
                        func = max
                    else:
                        func = min
                    oracle_complexity_counter = func(record['oracle_complexity_counter'], key=len)
                    sample_complexity_counter = func(record['sample_complexity_counter'], key=len)
                    iter_counter = func(record['iter_counter'], key=len)
                    epoch_counter = func(record['epoch_counter'], key=len)
                    total_oracle_complexity_counter = func(record['total_oracle_complexity_counter'], key=len)
                    total_sample_complexity_counter = func(record['total_sample_complexity_counter'], key=len)
                    total_iter_counter = func(record['total_iter_counter'], key=len)
                    total_epoch_counter = func(record['total_epoch_counter'], key=len)
                    #counter = total_oracle_complexity_counter[:data_xLimit]
                    counter = total_iter_counter[:data_xLimit]

                    if  "GS-GDA-B" in alg_name and "sin" in data_name:
                        step_back = 0 
                        if kappa in [50]:
                            step_back = 5000
                        if kappa in [100]:
                            step_back = 12345
                        if kappa in [1000]:
                            step_back = 0
                        if kappa in [10000]:
                            step_back = -240000
                        
                        for i, v in enumerate(counter):
                            counter[i]+= step_back

                    data_xLimit = min(data_xLimit, len(counter))

                    # load y-axis data
                    # valid_line_search = [i for i in range(len(record['acc'])) if len(record['acc'][i])>0]
                    # valid_line_search = [i for i in range(len(record['loss'])) if not np.isnan(record['loss'][i][data_xLimit-1])]
                    valid_line_search = pick_valid_line_search(record,alg_name)
                    print(valid_line_search)

                    acc = record['acc']
                    acc = [acc[i][:data_xLimit] for i in valid_line_search]
                    loss = [record['loss'][i][:data_xLimit] for i in valid_line_search]
                    error = [[1 - ele[i] for i in range(len(acc[0]))] for ele in acc]
                    lr_x = record['lr_x']
                    lr_y = record['lr_y']
                    lr_x = [lr_x[i][:data_xLimit] for i in valid_line_search]
                    lr_y = [lr_y[i][:data_xLimit] for i in valid_line_search]
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
                        shadowplot(counter, norm_sqaure_full_grad_x, label_input=alg_name, alpha=0.2, center=C, is_log=is_log,
                                is_var=True, alg_name=alg_name)
                    elif plot_part == 'y':
                        shadowplot(counter, norm_sqaure_full_grad_y, label_input=alg_name, alpha=0.2, center=C, is_log=is_log,
                                is_var=False, alg_name=alg_name)
                    elif plot_part == 'z':
                        shadowplot(counter, norm_sqaure_full_grad_z, label_input=alg_name, alpha=0.2, center=C, is_log=is_log,
                                is_var=False, alg_name=alg_name, is_last=is_last)
                    elif plot_part == 'acc':
                        shadowplot(counter, error, label_input=alg_name, alpha=0.2, center=C, is_log=is_log, is_var=True,
                                alg_name=alg_name, is_last=is_last)
                    elif plot_part == 'loss':
                        shadowplot(counter, loss, label_input=alg_name, alpha=0.2, center=C, is_log=is_log, is_var=True,
                                alg_name=alg_name, is_last=is_last)
                    elif plot_part == 'lr_x':
                        shadowplot(counter, lr_x, label_input=alg_name, alpha=0.2, center=C, is_log=is_log, is_var=True,
                                alg_name=alg_name)
                    elif plot_part == 'lr_y':
                        shadowplot(counter, lr_y, label_input=alg_name, alpha=0.2, center=C, is_log=is_log, is_var=True,
                                alg_name=alg_name)


            plt.xlabel("Number of gradient calls", fontsize=15)

            if plot_part == 'x':
                plt.ylabel(r"$\frac{||\nabla_x\mathcal{L}(x_k,y_k)||^2}{||\nabla_x\mathcal{L}(x_0,y_0)||^2}$", fontsize=15)
            elif plot_part == 'y':
                plt.ylabel(r"$\frac{||\nabla_y\mathcal{L}(x_k,y_k)||^2}{||\nabla_y\mathcal{L}(x_0,y_0)||^2}$", fontsize=15)
            elif plot_part == 'z':
                if not is_last:
                    plt.ylabel(r"$\|\nabla\mathcal{L}(x,y)||^2$", fontsize=15)
                    plt.ylabel(r"$\frac{||\nabla\mathcal{L}(x_k,y_k)||^2}{||\nabla\mathcal{L}(x_0,y_0)||^2}$", fontsize=20)
                else:
                    plt.ylabel(r"$\min_{i=0,1,...,k} \frac{||\nabla\mathcal{L}(x_i,y_i)||^2}{||\nabla\mathcal{L}(x_0,y_0)||^2}$", fontsize=20)
            elif plot_part == 'acc':
                plt.ylabel(r"Train Error", fontsize=15)
            elif plot_part == 'loss':
                plt.ylabel(r"Primal value", fontsize=15)
            elif plot_part == 'lr_x':
                plt.ylabel(r"Step size $\tau$", fontsize=15)
            elif plot_part == 'lr_y':
                plt.ylabel(r"Step size $\sigma$", fontsize=15)

            # set x,y range here
            # plt.ylim(1e-2,)
            plt.xlim(0,plot_xLimit)

            # set personalized axis scale here
            if plot_part == 'x':
                plt.yscale('log')
            elif plot_part == 'y':
                plt.yscale('log')
            elif plot_part == 'z':
                plt.yscale('log')
                # if kappa == 10:
                    #plt.ylim(1e-7,1.1)
            elif plot_part == 'acc':
                plt.ylim(0, 0.6)
            elif plot_part == 'loss':
                plt.yscale('log')
            elif plot_part == 'lr_x':
                plt.yscale('log')
            elif plot_part == 'lr_y':
                plt.yscale('log')
            
            if kappa in [5,10,20,50]:
                plt.xlim(0,1e4)
            elif kappa in [100]:
                plt.xlim(0,2e4)
            elif kappa in [1000]:
                plt.xlim(0,3e4)
            elif kappa in [10000]:
                plt.xlim(0,3e4)

            if kappa in [5,10,20,50] and "sin" in data_name:
                plt.ylim(1e-10,1.0001)

            # plt.xscale('log')
            if is_log:
                ax.set_yticklabels([round(np.exp(y) + C, 2) for y in ax.get_yticks()], fontsize=10)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax.xaxis.offsetText.set_visible(True)
            plt.grid()
            # set title here
            # plt.title('Qudradic_Bilinear_Obj',fontsize = 15)


            if plot_part == 'x':
                plt.legend(fontsize=10, loc='upper right')
            elif plot_part == 'y':
                plt.legend(fontsize=10, loc='upper right')
            elif plot_part == 'z':
                if "sin" in data_name:
                    plt.legend(fontsize=10, loc='lower left')
                if mu_y == 0.0001:
                    plt.legend(fontsize=10, loc='lower right')
                else:
                    plt.legend(fontsize=10, loc='lower left')
                plt.legend(fontsize=10, loc='upper right')
            elif plot_part == 'acc':
                plt.legend(fontsize=10, loc='upper right')
            elif plot_part == 'loss':
                plt.legend(fontsize=10, loc='upper right')
            elif plot_part == 'lr_x':
                plt.legend(fontsize=10, loc='lower right')
            elif plot_part == 'lr_y':
                plt.legend(fontsize=10, loc='lower right')
                
            # set label size here
            plt.rc('xtick', labelsize=15)
            plt.rc('ytick', labelsize=15)
            
            data_name_tmp = list(data_name)
            for i in range(len(data_name_tmp)):
                if data_name_tmp[i] == '.':
                    data_name_tmp[i] = '_'
            if is_last:
                name = f'./figure/{"".join(data_name_tmp)}_{plot_part}_last.pdf'
            else:
                name = f'./figure/{"".join(data_name_tmp)}_{plot_part}.pdf'
                
            plt.savefig(name, bbox_inches='tight', facecolor='w', dpi=150)
