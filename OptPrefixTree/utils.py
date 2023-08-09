from scipy.stats import binom
import scipy.stats as stats
import math
import random
from collections import Counter

def selectData(data_list, weights = "uniform"): 
        
    # Count the frequency of each element
    element_counts = Counter(data_list)
    element_counts_items = list(element_counts.keys())
    
    # Calculate cumulative probabilities
    total_elements = len(data_list)
    if weights == "uniform":
        weights_probabilities = [1 / len(element_counts_items) for element in element_counts_items]
    if weights == "frequency":
        weights_probabilities = [element_counts[element] / total_elements for element in element_counts_items]

    sampled_element = random.choices(element_counts_items, weights=weights_probabilities)[0]
    return sampled_element


def PruneHH(D, estimated_frequency_D, tau_0, FPR, sigma, eta):
    """_summary_

    Args:
        D (list): query set, aggregated data domain
        estimated_frequency_D (list): estimated frequency after aggregation
        tau_0 (float): initialization of threshold
        FPR (float): ratio of expected false positives to total number of bins
        sigma (float): aggregated noise standard deviation
        eta (float): step size
    """
    E = 1-stats.norm.cdf(tau_0* sigma) 
    P_prefixlist = []
    for indx, item in enumerate(D):
        if estimated_frequency_D[indx] > tau_0*sigma:
            P_prefixlist.append(item)
    while FPR < E*len(D)/len(P_prefixlist):
        E = eta*E
        P_prefixlist = []
        for indx, item in enumerate(D):
            if estimated_frequency_D[indx] >  stats.norm.ppf(1-E, loc=0):
                P_prefixlist.append(item)
    return P_prefixlist


def frequency_estimation(D, eps_l, opt=False):
    """_summary_

    Args:
        eps_l (_type_): local privacy parameter
        D (_type_): Dataset
        opt (bool, optional): Defaults to True.
    """
    # f_w for d occurence in D
    n = len(D)
    counter = Counter(D)
    unique_elements = list(counter.keys())
    
    if opt:
        a_1 = 1/2
        a_0 = 1/(math.exp(eps_l) + 1)
    else:
        a_0 = 1/(math.exp(eps_l)+1)
        a_1 = 1-a_0
    estimated_frequency_D = []
    variances = []
    for d in unique_elements:
        f_w = counter[d]
        binom_successes_f_w = binom.rvs(n=f_w, p=a_1, size=1)
        binom_successes_n_minus_f_w = binom.rvs(n=n - f_w, p=a_0, size=1)
    
        tilde_f_w = (binom_successes_f_w + binom_successes_n_minus_f_w - n * a_0) / (a_1 - a_0)
        estimated_frequency_D.append(tilde_f_w)
        var = tilde_f_w*(1-a_0-a_1)/(a_1 - a_0) + n*a_0*(1-a_0)/(a_1-a_0)**2
        variances.append(var)
        
        
    return unique_elements, estimated_frequency_D, variances


