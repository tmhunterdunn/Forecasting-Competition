import pandas as pd
import numpy as np
from scipy.stats import norm
from statsmodels.tsa.stattools import acf, acovf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt

def weak_wn_acf(returns, lags=40, alpha=0.95):
    n = len(returns)
    alpha = 0.95
    gamma_0 = acovf(returns)[0]
    returns_sq = np.array(returns)**2
    sigma_hat_p = ((1/n)*np.array([returns_sq[:n-h] @ returns_sq[h:] for h in range(1,n)])/(gamma_0**2))**0.5
    
    upper = sigma_hat_p*norm.ppf(alpha)/(n**0.5)
    upper = np.concatenate([ [float('NaN')], upper])
    lower = -sigma_hat_p*norm.ppf(alpha)/(n**0.5)
    lower = np.concatenate([ [float('NaN')], lower])

    plot_acf(returns, lags=lags, alpha=1-alpha)
    plt.plot(upper[:lags+1])
    plt.plot(lower[:lags+1])
    plt.legend(['Strong WN CI', 'Auto Correlation', 'Weak WN CI Upper', 'Weak WN CI Lower'])
    plt.show()
    
    
    
def fourier_extrapolation(x, n_predict):
    """x: the fourier coeffs
    n_predict: length of extrapolation
    """
    
    n = x.size
    n_harm = (x!=0).sum()                     # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    f = np.fft.fftfreq(n)              # frequencies
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i: np.absolute(x[i]), reverse=True)
 
    t = np.arange(0, n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x[i]) / n   # amplitude
        phase = np.angle(x[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig
