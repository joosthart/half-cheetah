from multiprocessing import Pool
import os
from src.mcpg.agent import *
import matplotlib.pyplot as plt

# plt.style.use('thesis.mplstyle')

def plot_mcpg(mean_str, std_str, rew_str, params, savepath, seeds):
    """ Plots the Figures of the MCPG algorithms
    """
    means = np.empty((params[-1], len(seeds)))
    for idx, seed in enumerate(seeds):
        loaded =  np.load(savepath + mean_str.format(params[-1], 
                                                       params[0], 
                                                       params[1], 
                                                       np.bool(params[2]), 
                                                       np.int(params[3]), 
                                                       np.int(seed)))
    
        means[:,idx] = loaded
        
    std  = np.std(means, axis = -1)
    mean = np.mean(means, axis = -1)
    
    fig, ax = plt.subplots()
    ax.plot(mean, label = 'Mean')
    ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.5, label='Standard Deviation')
    
    c = ['r', 'g', 'orange']
    
    for idx, seed in enumerate(seeds):
        loaded =  np.load(savepath + mean_str.format(params[-1], 
                                                       params[0], 
                                                       params[1], 
                                                       np.bool(params[2]), 
                                                       np.int(params[3]), 
                                                       np.int(seed)))
        
        if idx == 2:
            plt.plot(loaded, c = c[idx], label = 'Rolling Mean Reward'.format(seed))
        else:
            plt.plot(loaded, c = c[idx])
        
        loaded =  np.load(savepath + rew_str.format(params[-1], 
                                                       params[0], 
                                                       params[1], 
                                                       np.bool(params[2]), 
                                                       np.int(params[3]), 
                                                       np.int(seed)))
        if idx == 2:
            plt.plot(loaded, c = c[idx], label = 'Episodal Reward', alpha = 0.5)
        else:
            plt.plot(loaded, c = c[idx], alpha = 0.5)
        
    
    plt.text(0.8*len(mean), np.amax(mean), '')    
    ax.set_xlim(0, len(mean))
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rewards')
    ax.legend(loc = 0)
    ax.set_xlim(xmax = 100)
    plt.tight_layout()
    plt.savefig(savepath + "Simulation_mcpg_lr{}_gamma{}_Normalized{}_N_hidden_layers{}.pdf".format(params[0], 
                                                                                                    params[1], 
                                                                                                    np.bool(params[2]), 
                                                                                                    np.int(params[3])), 
                dpi=300)
    plt.close()


def make_figures_mcpg():
    """ Generetes figures shown in the report
    """
    
    #initialize file names
    direc = './output/mcpg/'
    
    #Get figure of longer episode number runs
    mean_str = 'running_mean{}e_lr{}_gamma{}_normalize{}_hiddenlayers{}_seed{}.npy'
    std_str  = 'running_std{}e_lr{}_gamma{}_normalize{}_hiddenlayers{}_seed{}.npy'
    rew_str  = 'last_results{}e_lr{}_gamma{}_normalize{}_hiddenlayers{}_seed{}.npy'
    plot_mcpg(mean_str, std_str, rew_str, [5e-3, 0.999, True, 2, 1000], direc, [41, 42, 43])
#     plot_mcpg(mean_str, std_str, rew_str, [0.01, 0.999, True, 2, 1000], direc, [41, 42, 43])

    
def multi_mcpg(seed):
    
    lr, gamma, normalize, n_hidden, seed = [5e-3, 0.999, True, 2, seed]
#     lr, gamma, normalize, n_hidden, seed = [0.01, 0.999, True, 2, seed]

    last_results, running_mean, running_loss, running_std = MDP(lr, gamma, normalize, n_hidden, seed, 1000)

    #save
    np.save('./output/mcpg/last_results{}e_lr{}_gamma{}_normalize{}_hiddenlayers{}_seed{}.npy'.format(len(last_results), 
                                                                              lr,
                                                                              gamma,
                                                                              normalize,
                                                                              n_hidden,
                                                                              seed), last_results)
    np.save('./output/mcpg/running_mean{}e_lr{}_gamma{}_normalize{}_hiddenlayers{}_seed{}.npy'.format(len(last_results),
                                                                              lr,
                                                                              gamma,
                                                                              normalize,
                                                                              n_hidden,
                                                                              seed), running_mean)
    np.save('./output/mcpg/running_loss{}e_lr{}_gamma{}_normalize{}_hiddenlayers{}_seed{}.npy'.format(len(last_results),
                                                                              lr,
                                                                              gamma,
                                                                              normalize,
                                                                              n_hidden,
                                                                              seed), running_loss)
    np.save('./output/mcpg/running_std{}e_lr{}_gamma{}_normalize{}_hiddenlayers{}_seed{}.npy'.format(len(last_results),
                                                                              lr,
                                                                              gamma,
                                                                              normalize,
                                                                              n_hidden,
                                                                              seed), running_std)
def run_all_experiments(n_cores = 3):
    """Run all done experiments.
    
    Args:
        n_cores(int, optional): number of cores used for multiprocessing
    """
    seeds = [41,42,43]
    pool = Pool(n_cores)
    pool.map(multi_mcpg, seeds)
    pool.close()

    make_figures_mcpg()    
                                                                
if __name__ == '__main__':
    run_all_experiments()