"""
Operations for diffraction image reduction

Includes:

"""
import os
from pathlib import Path
import numpy as np
import scipy
import scipy.io
from scipy.optimize import curve_fit
from scipy import integrate, signal
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad
import pandas as pd
import fabio
import re

import matplotlib.pyplot as plt

import pyFAI
from pyFAI.multi_geometry import MultiGeometry
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator as AzInt

from .peakShapes import voigtFn, gaussFn
from .utils import folder_select

# To-Do: make type alisases (list[Path's])


def load_image(path: Path=Path('')) -> np.ndarray:
    """Open image based on file suffix.  
    """
    if path.suffix in ['.tif', '.tiff']:
        # open tiff image
        im = fabio.open(path)
        # input image object into a numpy array
        imArray = im.data
    elif path.suffix in ['.raw']:
        # extract raw file
        im = open(path, 'rb')
        arr = np.fromstring(im.read(), dtype='int32')
        im.close()
        
        #raw requires prompting for dimensions, hard code for now
        arr.shape = (195, 487)
        imArray = np.array(arr)
        
    return imArray

center = (183, 245)
det = pyFAI.detectors.Pilatus100k()
sampleExpInfo = {'dist': 0.55, # meters
                'poni1': center[0] * det.pixel1,
                'poni2': center[1] * det.pixel2,
                'detector': det,
                'wavelength': 7.2932E-11,
                'orientation': 'horizontal'
                }

def create_scan(imgList: list, specPath: Path, 
                expInfo: dict=sampleExpInfo) -> (MultiGeometry, list):
    """Create multigometry object containing images and coordinates 
    """
    spec = pd.read_csv(specPath)
    spec = spec.rename(columns=lambda x: x.strip())
    ais = []
    imgs = []

    for im in imgList:
        dat = load_image(im)    

        if expInfo['orientation'].lower() == "vertical":
            imgs.append(np.rot90(dat, k=1))
        elif expInfo['orientation'].lower() == 'horizontal':
            imgs.append(dat)
        
        scanNo = int(re.search(r'(\d{4})(\.raw)', str(im)).group(1))

        intInfo = {x: expInfo[x] for x in ['poni1', 'poni2', 'dist',
                                            'detector', 'wavelength']}

        ai = AzInt(**intInfo, 
                    rot2=np.pi/180*float(spec['TwoTheta'][scanNo]),
                    rot3=-np.pi/2)
        ais.append(ai)
    ##############################################################################
    ##############################################################################
    ##############################################################################
    mg = MultiGeometry(ais, unit="2th_deg", 
                        radial_range=(16, 28), azimuth_range=(-20, 20))
    ##############################################################################
    ##############################################################################
    ##############################################################################


    return mg, imgs

def generate_mask(img: list, xmin: int=0, xmax: int=-1, 
                    ymin: int=0, ymax: int=-1) -> np.ndarray:
    """Generates a mask the size of the image 
        where 1's are invaid, 0's are valid.  
    """
    mask = np.ones(np.shape(img))
    mask[ymin:ymax, xmin:xmax]
    return mask

def save_qchi(mg: MultiGeometry, imgs: list, mask: np.ndarray,
                spath: Path, template: str):
    # Create directory
    os.makedirs(spath, exist_ok=True)

    _, ax = plt.subplots(1,1)

    # Integrate image
    res2d = mg.integrate2d(imgs, 1000,360, lst_mask=mask)
    # gives 3 arrays.  First is intensities with shape (arg2, arg1)  
    # Second is 2th range, third is azimuthal range

    ax.pcolormesh(res2d[1], res2d[2], np.log(res2d[0]), cmap = 'viridis')
    plt.rc('text')
    plt.rc('font')

    ax.set_xlim(0,65)
    ax.set_xlabel('Radial Angle (2theta)')
    ax.set_ylabel('Azumuthal Angle (chi)')

    plt.savefig(spath + template + '_figures.png')

    scipy.io.savemat(spath + template + '_Qchi.mat', 
                     {'Q':res2d[1], 'chi':res2d[2], 'cake':res2d[0]})

def save_Itth(mg: MultiGeometry, imgs: list, mask: np.ndarray,
                spath: Path, template: str=''):
    # Create directory
    os.makedirs(spath, exist_ok=True)
    
    _, ax = plt.subplots(1,1)
    
    int1d = mg.integrate1d(imgs, 10000, lst_mask=mask)
    ax.plot(int1d[0], int1d[1])
    ax.set_ylabel('Intensity')
    ax.set_xlim(0,65)

    int1List = list(int1d)
    int1df = pd.DataFrame(data=int1List[1], index=int1List[0])
    int1df.to_csv(spath + template + '_1D.csv')

def save_dict(data: dict, spath: Path, template: str=''):
    """Save data to spath+template, append if already exists
    """
    if os.path.exists(spath+template):
        df = pd.read_csv(spath+template+'_params.csv')
        dfnew = pd.DataFrame(data)
        df = df.append(dfnew)
        df.to_csv(spath+template+'_params.csv')
    else:
        # create new file
        pd.DataFrame(data).to_csv(spath+template+'_params.csv')

def summarize_params(csv_path: Path, 
                    search_str: str='', save_template: str=''):
    """Generate summary csv unique to Alan's plans. 

    pick out peaks with x-locations closest to provided tth locations
    append to area_summary.csvname   """
    csvList = folder_select(csv_path, search_str)
    result = None
    for f in csvList:
        df = pd.read_csv(f)
        df = df.rename(columns={df.columns[0]: 'param'})

        # whacky sorting thing
        df = df.set_index('param')
        df = df.sort_values(by='x0', axis=1)
        df.columns = [f'curve {j}' for j in range(len(df.columns))]
        df = df.rename_axis('param').reset_index()

        # Flatten and group.  Kind of a 'join on'
        melt = pd.melt(df, id_vars=df.columns[0], 
                        value_vars=df.columns[1:], value_name=f.stem)

        if result is None:
            result = melt
        else:
            result = result.merge(melt, on=('param', 'variable'))

    result.to_csv(Path(csv_path) / save_template)

def save_curve_fit(x, y, params: dict, spath: Path, 
                    template: str='', 
                    peakShape: str='voigt'):
    """Plot curves vs data (x,y) for given curve parameters
    """
    if peakShape == 'Voigt':
        func = voigtFn
    elif peakShape == 'Gaussian':
        func = gaussFn
    else: 
        print('no peak shape chosen')

    ncurves = len(params)
    plt.figure(figsize=(8,8))

    popt = []
    for j, (_, l) in zip(range(ncurves), params.items()):
        temp = list(l.values())
        popt += temp
        plt.plot(x, func(x, *l.values()), '.', alpha=0.5, 
                label='opt. curve {:.0f}'.format(j))


    plt.plot(x, y, marker='s', color='k', label='data')
    plt.plot(x, func(x, *popt), 
            color='r', label='combined data')
    
    plt.legend()

    plt.savefig(spath + template + 'peakAt_' + 
                    '{:.3f}'.format(x[np.where(y==np.max(y))[0][0]]) + '.png')
    plt.close()        

def integrate_1d_mg(mg: MultiGeometry, imgs: list, 
                    mask: np.ndarray=None) -> np.ndarray:
    """Perform 1D integration on many images
    first axis is (2th or q), second is intensity
    """
    print(np.shape(imgs))
    int1d = mg.integrate1d(imgs, 10000, lst_mask=mask)
    
    return np.array(int1d)

def restrict_range(x, y, xRange: list=None, 
                    yRange: list=None) -> (np.ndarray, np.ndarray):
    """Return the subset of x, y that falls within xRange and yRange

    """

defaultFitOpts = {'peakShape': 'voigt', # ('voigt', 'gaussian')
                    'fitMode': 'fixed',
                    'numCurves': 4,
                    }

def fit_peak(x: np.ndarray=np.ones(5,), y: np.ndarray=np.ones(5,),
                peakShape: str=defaultFitOpts['peakShape'], 
                fitMode: str=defaultFitOpts['fitMode'],
                numCurves: int=defaultFitOpts['numCurves']
            ) -> (dict, list):
    """Fit peak segment with specified peak shape and method
    return dictionary with peak information
    finalParams: all curve info
    FWHM: calculated FWHM's for each curve
    """

    xRange = np.ptp(x)
    maxInd = np.where(y == np.max(y))[0][0]

    # Initialize guess and fitting limits for each function
    if peakShape == 'Voigt':
        func = voigtFn
        # x0 and I values to be replaced during loop
        guessTemp = [0, np.min(y), 0,
                        xRange / 10, xRange / 10 ]
        # x bounds to be replaced
        # [x0, y0, I, alpha, gamma]
        boundLowTemp = [x[0], np.min(y), 0., 0., 0.]
        boundUppTemp = [x[-1], np.inf, np.inf, xRange, xRange]

        boundLowerPart = [x[0]-0.05*xRange, np.min(y), 0, 0, 0]
        boundUpperPart = [x[-1]+0.05*xRange, np.inf, np.inf, 
                            xRange, xRange]
    elif peakShape == 'Gaussian':
        func = gaussFn     
        # x0 and I values to be replaced during loop
        guessTemp = [0, np.min(y), 0,
                        xRange / 10]
        # x bounds to be replaced
        # [x0, y0, I, sigma]
        boundLowTemp = [x[0]-0.05*xRange, np.min(y), 0., 0.]
        boundUppTemp = [x[-1]+0.05*xRange, np.inf, np.inf, xRange]

        boundLowerPart = [x[0], np.min(y), 0, 0]
        boundUpperPart = [x[-1], np.inf, np.inf, xRange]       

    # Initialize guess params and bounds for number of curves
    boundUpper = []
    boundLower = []
    guess = []
        
    # init fit array to compare residual 
    fit = np.ones_like(y) * np.min(y)
    
    # parameter array: [x0, y0, Intensity, alpha, gamma]
    # set up guesses
    curveCnt = 0
    resid = fit - y
    # To-do: figure out better way to zero shift
    errorCurr = np.mean(np.absolute(resid) / (y+1))
    print('Peak at {0}, start iteration with error = {1}'.format(x[maxInd], errorCurr))
    while curveCnt < numCurves: # and errorCurr > 0.001: 
        # place peak at min residual
        xPosGuess = x[np.argmin(resid)]
        guessTemp[0] = xPosGuess
        guessTemp[2] = np.max(resid) - np.min(resid)
            
        # Deal with edge cases.. 
        xPosLow = xPosGuess - 0.01*xRange
        if xPosLow < x[0]: xPosLow = x[0]
        xPosHigh = xPosGuess + 0.01*xRange
        if xPosHigh > x[-1]: xPosHigh = x[-1]
            
        # Update temp bounds to be close to position guess
        boundLowTemp[0] = xPosLow
        boundUppTemp[0] = xPosHigh

        boundTemp = tuple([boundLowTemp, boundUppTemp])
        try: # Fit according to residual
            poptTemp, pcovTemp = curve_fit(func, x, -resid, 
                                        bounds=boundTemp, p0=guessTemp)  
            #print('Fit to residual at {0}'.format(xPosGuess))
        except RuntimeError as e:
            print(e) 
            poptTemp = guessTemp

        # Check to see if error decreased
        guessHold = guess + list(poptTemp)
        fit = func(x, *guessHold)
        resid = fit - y
        errorNew = np.mean(np.absolute(resid) / (y+1))
        # if np.absolute(errorCurr - errorNew) < 0.0001:
        #     print('no change in error: {}'.format(errorNew))

        #     if curveCnt == 0: # if first peak does not change error
        #         # build guess real guess array, update fit
        #         guess = guessHold

        #         # concatenate lists for bounds for real fit
        #         boundLower += boundLowerPart 
        #         boundUpper += boundUpperPart

        #         # Combine bounds into tuple for input
        #         bounds = tuple([boundLower, boundUpper])
           
        #     # break #end iteration

        # build guess real guess array, update fit
        guess = guessHold

        # concatenate lists for bounds for real fit
        boundLower += boundLowerPart 
        boundUpper += boundUpperPart

        # Combine bounds into tuple for input
        bounds = tuple([boundLower, boundUpper])
       
        # Calculate residual, increment, and print error information 
        errorCurr = errorNew 
        print('Peak at {0}, iteration {1}: error = {2}'.format(x[maxInd], 
                                                    curveCnt, errorCurr))
        curveCnt+=1

    ####################################### Now fit whole peak with acq curves 
    # Fit full curve, refining guesses
    try:
        # Curve fit function call using guess and bounds
        popt, pcov = curve_fit(func, x, y, 
                                    bounds=bounds, p0=guess)
    except RuntimeError as e:
        print(e) 
        popt = np.array(guess)
       

    # Calculate FWHM, area for each peak fit
    if peakShape == 'Voigt':
        FWHM = []
        names = ['x0', 'y0', 'I', 'alpha', 'gamma']
        c0 = 2.0056
        c1 = 1.0593
        for i in range(0, len(popt), 5): # grab fwhm for each peak
            fg = 2*popt[i+3]*np.sqrt(2*np.log(2))
            fl = 2*popt[i+4]
            phi = fl / fg
            FWHM.append(fg * (1-c0*c1 + np.sqrt(phi**2 + 2*c1*phi + (c0*c1)**2)))

    elif peakShape == 'Gaussian':
        FWHM = []
        names = ['x0', 'y0', 'I', 'alpha', 'gamma']
        for i in range(0, len(popt), 4): # grab fwhm for each peak
            FWHM.append(2*popt[i+3]*np.sqrt(2*np.log(2)))

    ###########################################################################
    ### Organizing output
    ###########################################################################
    
    # Organize final parameters into dict
    curveParams = {}
    derivedParams = {}
    for j in range(curveCnt):   # Plot each individual curve
        L = int(0 + j * len(popt) / curveCnt)  # Sep popt array
        R = int((j+1) * len(popt) / curveCnt)

        area, err = quad(func, x[0], x[-1], tuple(popt[L:R]))

        tempd = {n:v for (n,v) in zip(names,popt[L:R])}
        curveParams[f'curve {j}'] = tempd
        # Init derived curve  sub-dict
        derivedParams[f'curve {j}'] = {}
        derivedParams[f'curve {j}']['FWHM'] = FWHM[j]
        derivedParams[f'curve {j}']['area'] = area
        derivedParams[f'curve {j}']['area_err'] = err
        derivedParams[f'curve {j}']['x0'] = curveParams[f'curve {j}']['x0']
    
    return curveParams, derivedParams