import numpy as np
from scipy.special import wofz

def voigtFn(x, *params):
    """
    Return Voigt line shape centered at x0 with intensity I
    alpha: Lorentzian comp HWHM
    gamma: Gaussian comp HWHM
    I: Intensity
    x0, y0: Offsets
    params = [x0, y0, I, alpha, gamma] * numCurves
    """
    result = 0
    # grab one set of parameters for each curve
    for i in range(0,len(params),5):
        x0    = params[i]
        y0    = params[i+1]
        I     = params[i+2]
        alpha = params[i+3]
        gamma = params[i+4]
    
        sigma = alpha / np.sqrt(2 * np.log(2))

        result = result + (I * np.real(wofz(((x-x0) + 1j*gamma)/sigma/np.sqrt(2))) 
                            / sigma / np.sqrt(2*np.pi) + y0)
    
    return result


def gaussFn(x, *params):
    """
    Return Gaussian line shape centered at x0 with intensity I
    I: Intensity
    x0, y0: Offsets
    sigma: Gaussian comp stdev
    parames = [x0, y0, I, sigma] * numCurves
    """
    result = 0
    # grab one set of parameters for each curve
    for i in range(0,len(params),4):
        x0    = params[i]
        y0    = params[i+1]
        I     = params[i+2]
        sigma = params[i+3]
        
        result += I * np.exp(-(x-x0)**2 / (2*sigma**2)) + y0
    
    return result
    
def lorentzFn(x, *params):
    """
    Return Lorentzian line shape centered at x0 with intensity I
    I: Intensity
    x0, y0: Offsets
    sigma: Gaussian comp stdev
    parames = [x0, y0, I, sigma] * numCurves
    """
