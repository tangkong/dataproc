"""
Operations for diffraction image reduction

Includes:

"""
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.polynomial.chebyshev import chebval
import scipy
import scipy.io
from scipy.optimize import curve_fit, basinhopping
from scipy import integrate, signal
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import quad

import pandas as pd
import fabio
import re

import matplotlib.pyplot as plt

import pyFAI
import pyFAI.detectors as dets
from pyFAI.multi_geometry import MultiGeometry
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator as AzInt

from .peakShapes import voigtFn, gaussFn
from .utils import folder_select

# To-Do: make type alisases (list[Path's])
center = (183, 245)
det = pyFAI.detectors.Pilatus100k()
sampleExpInfo = {'dist': 0.55, # meters
                'poni1': center[0] * det.pixel1,
                'poni2': center[1] * det.pixel2,
                'detector': det,
                'wavelength': 7.2932E-11, # meters?..
                'orientation': 'horizontal'
                }

def load_image(path: Path=Path('')) -> np.ndarray:
    """load_image loads images based on file extension
        
    :param path: path to single image file, defaults to Path('')
    :type path: Path, optional
    :return: image array data
    :rtype: np.ndarray
    """
    try:
        path = Path(path)
        # check if path exists (path.exists())
    except:
        print('could not verify path')
        im = np.array([0])

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

def parse_calib(filename):
    """parse_calib Parses calibration from wxDiff
    
    :param filename: [description]
    :type filename: [type]
    :return: [description]
    :rtype: [type]
    """
    file=open(filename,'r')
    data = []
    with file as inputfile:
        for line in inputfile:
            data.append(line.strip().split('\n'))
    bcenter_x = float(data[6][0][10:])
    bcenter_y = float(data[7][0][10:])
    detector_dist = float(data[8][0][12:])
    detect_tilt_alpha = float(data[9][0][18:])
    detect_tilt_delta = float(data[10][0][18:])
    wavelength = float(data[11][0][11:])
    return detector_dist, detect_tilt_alpha, detect_tilt_delta, wavelength, bcenter_x, bcenter_y

def db_to_pattern_array(hdr, cfg: dict, img_key='pilatus1M_image'):
    # returns dataframe with keys like: [s_stage.px, s_stage.px, pilatus1M_image]
    df = hdr.table(fill=True)

    # iterate through images, integrate, generate 1D patterns, add to df
    # Don't modify something you're iterating over, in general.  Apply instead
    det = getattr(dets, cfg['expInfo']['detector'])()
    def _integrate(row, cfg=cfg, det=det):
        img = row[cfg['dbInfo']['imgkey']][0]
        einfo = cfg['expInfo']
        p = pyFAI.AzimuthalIntegrator(wavelength=einfo['wavelength'])
        p.setFit2D(einfo['dist'], einfo['center1'], einfo['center2'],  
                    einfo['tilt'], einfo['rot'], det.pixel1, det.pixel2)
        return p.integrate1d(img, 1000)

    df['qI_data'] = df.apply(_integrate, axis=1)
    
    # format into [x, y, len(pattern)] array.  empty spots are 0
    # assume uniform spacing
    pattern_length = len(df['qI_data'][1])
    xlocs = df['s_stage_px'].unique() #values may not be round, use actual locs
    xlocs.sort()
    ylocs = df['s_stage_py'].unique()
    ylocs.sort()
    patterns = np.zeros((len(xlocs), len(ylocs), pattern_length))

    for i, xl in enumerate(xlocs): # indices
        for j, yl in enumerate(ylocs):
            sub_df = df[(df['s_stage_px']==xl) & (df['s_stage_py']==yl)]
            if len(sub_df) > 0:
                patterns[i, j, :] = sub_df['qI_data'].item() # TODO verify

    return patterns

def files_to_pattern_array(search_string: str, csv_path: Path, 
                            img_path: Path, export_path: Path, 
                            exp_cfg: dict=sampleExpInfo):
    try:
        df = pd.read_csv(csv_path.glob(search_string))
    except:
        print('string does not match any csvs in {csv_path}')
        return
    
    img_list = list(img_path.glob(search_string))
    # coalate image and position data
    for ip in img_list:
        im = load_image(ip)
        ai  = create_single(im, exp_cfg)
        q, I = ai.integrate1d(im, npts=1000) # same config works for each image
        
        
    
    patterns = np.zeros((5,5,5))
    return patterns

def create_single(expInfo: dict=sampleExpInfo) -> np.ndarray:
    """create_single [summary]
    
    Creates single scan, look to see if normal azimuthal integrator 
    operates similarly to multigeometry for save_qchi, save_Itth
    """
    det = getattr(dets, expInfo['detector'])

    p = pyFAI.AzimuthalIntegrator(wavelength=expInfo['wavelength'], det=det)
    p.setFit2D(expInfo['dist'], expInfo['center1'], expInfo['center2'],  
                    expInfo['tilt'], expInfo['rot'], det.pixel1, det.pixel2)

    return p

def create_scan(imgList: list, specPath: Path, 
                expInfo: dict=sampleExpInfo) -> Tuple[MultiGeometry, list]:
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

    plt.close()
    
def save_dict(data: dict, spath: Path, template: str=''):
    """Save data to spath+template, append if already exists
    """
    os.makedirs(spath, exist_ok=True)

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
                    yRange: list=None) -> Tuple[np.ndarray, np.ndarray]:
    """Return the subset of x, y that falls within xRange and yRange

    """

def bkgd_sub(x, y, downsample_int: int=50, 
            print_plot: bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract background with modified chebyshev polynomial
    """
    print("downsample:", downsample_int)
    def downsamp(size, x, y):
        """Downsample data based on size parameter.  Return downsampled data
        """
        yNew = y[::size] 
        xNew = x[::size] 
        
        return xNew, yNew
        
    def chebFunc(x, *params):
        """
        Modified chebyshev function with 1/x 
        """
        params = params[0]
        y = chebval(x, params[:4])
        E = params[4]
        y = y + E/x
        return y


    def objFunc(*params):
        """
        implement cost function for modified chevbyshev function
        varies cost to emphasize low Q region
        Scoping pulls trimmed x, y data from surrounding scope... this is bad

        return: cost of fit 
        """
        
        params = params[0]
        J = 0
        fit = chebval(downX, params[:4])
        E = params[4]
        fit = fit + E / downX

        for i in range(len(downY)):
            if downX[i] < 1:    # Treat low Q equally
                J = J + (downY[i] - fit[i])**4
            else:
                if downY[i] < fit[i]:
                    J = J + (downY[i] - fit[i])**4
                if downY[i] >= fit[i]:
                    J = J + (downY[i] - fit[i])**2
        return J
        
    # Create a sparse data set for fitting
    downX, downY = downsamp(downsample_int, x, y)
    
    # Remove points above median + 10% of range
    medianY = np.median(downY)
    rangeY = np.max(downY) - np.min(downY)
    downX = downX[downY <= (medianY + 0.1*rangeY)]
    downY = downY[downY <= (medianY + 0.1*rangeY)]
    
    # Re-append end values to try and fix boundary conditions
    downX = np.append(np.append(x[0:30:5], downX), x[-30:-1:5])
    downY = np.append(np.append(y[0:30:5], downY), y[-30:-1:5])

    x0 = [1,1,1,1,1]
    #appears to converge quickly, take 10 iterations rather than 100
    result = basinhopping(objFunc, x0, niter=100) 
    bkgd_sparse = chebFunc(x, result.x)
    # create function that interpolates sparse bkgd
    f = interp1d(x, bkgd_sparse, kind='cubic', bounds_error=False)

    # expressed background values
    bkgd1 = f(x)
    subDataY = y - bkgd1

    # Dump any nan values in all data
    finalDataY = subDataY[~np.isnan(subDataY)]
    finalDataX = x[~np.isnan(subDataY)]
    bkgd = bkgd1[~np.isnan(subDataY)]

    # Offset to make all values > 0 
    if np.min(subDataY) < 0:
        finalDataY = finalDataY + np.absolute(np.min(finalDataY))

    if print_plot:
        return finalDataX, finalDataY, bkgd1
    else: 
        return finalDataX, finalDataY

def bkgd_sub_GP(x, y, downsample_int: int=50): 
    pass

defaultFitOpts = {'peakShape': 'voigt', # ('voigt', 'gaussian')
                    'fitMode': 'fixed',
                    'numCurves': 4,
                    }

def fit_peak(x: np.ndarray=np.ones(5,), y: np.ndarray=np.ones(5,),
                peakShape: str=defaultFitOpts['peakShape'], 
                fitMode: str=defaultFitOpts['fitMode'],
                numCurves: int=defaultFitOpts['numCurves'],
                noise_estimate: np.ndarray = None,
                background: np.ndarray = None,
                stdratio_threshold: float = 2
            ) -> Tuple[dict, list]:
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
        boundLowTemp = [x[0], -np.max(y)/4, 0., 0., 0.]
        boundUppTemp = [x[-1], np.inf, np.inf, xRange/4, xRange/4]

        boundLowerPart = [x[0]-0.05*xRange, -np.max(y)/4, 0, 0, 0]
        boundUpperPart = [x[-1]+0.05*xRange, np.inf, np.inf, 
                            xRange/4, xRange/4]
    elif peakShape == 'Gaussian':
        func = gaussFn     
        # x0 and I values to be replaced during loop
        guessTemp = [0, np.min(y), 0,
                        xRange / 10]
        # x bounds to be replaced
        # [x0, y0, I, sigma]
        boundLowTemp = [x[0]-0.05*xRange, -np.max(y)/4, 0., 0.]
        boundUppTemp = [x[-1]+0.05*xRange, np.inf, np.inf, xRange/4]

        boundLowerPart = [x[0], -np.max(y)/4, 0, 0]
        boundUpperPart = [x[-1], np.inf, np.inf, xRange/4]       

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
    def keep_fitting(curveCnt):
        # fit more peaks as long as the residual RMSE is below-threshold
        # (or up to numCurves, if no noise estimate is given).
        if curveCnt >= numCurves:
            return False
        if curveCnt == 0:
            return True
        if noise_estimate is not None:
            ratio = np.sqrt(((resid**2).mean() / (noise_estimate**2).mean() ))
            print(ratio)
            if ratio < stdratio_threshold:
                return  False
        return True
    #while curveCnt < numCurves: # and errorCurr > 0.001: 
    while keep_fitting(curveCnt): # and errorCurr > 0.001: 
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

def bayesian_block_finder(x: np.ndarray=np.ones(5,), y: np.ndarray=np.ones(5,),):
    """bayesian_block_finder performs Bayesian Block analysis on x, y data.

    see Jeffrey Scargle's papers at doi: 10.1088/0004-637X/764/2/167

    :param x: array of x-coordinates
    :type x: numpy.ndarray
    :param y: array of y-values with same length as x
    :type y: numpy.ndarray
    """
    data_mode = 3
    numPts = len(x)
    if len(x) != len(y):
        raise ValueError('x and y are not of equal length')

    tt = np.arange(numPts)
    nnVec = []

    sigmaGuess = np.std(y[ y <= np.median(y)])
    cellData = sigmaGuess * np.ones(len(x))

    ncp_prior = 0.5


    ## To implement: trimming/choosing of where to start/stop
    ## To implement: timing functions

    cp = []
    cntVec = []

    iterCnt = 0
    iterMax = 10

    while iterCnt <= iterMax:
        best = []
        last = []

        for r in range(numPts):
            # y-data background subtracted
            sumX1 = np.cumsum(y[r::-1]) 
            sumX0 = np.cumsum(cellData[r::-1]) # sigma guess
            fitVec = (np.power(sumX1[r::-1],2) / (4 * sumX0[r::-1]))

            paddedBest = np.insert(best,0,0)
            best.append(np.amax( paddedBest + fitVec - ncp_prior ))
            last.append(np.argmax( paddedBest + fitVec - ncp_prior ) )

            # print('Best = {0},  Last = {1}'.format(best[r], last[r]))

        # Find change points by peeling off last block iteratively
        index = last[numPts-1]

        while index > 0:
            cp = np.concatenate( ([index], cp) )
            index = last[index-1]

        # Iterate if desired, to implement later
        iterCnt += 1
        break


    numCP = len(cp)
    numBlocks = numCP + 1

    rateVec  = np.zeros(numBlocks)
    numVec   = np.zeros(numBlocks)

    cptUse = np.insert(cp, 0, 0)

    #print('cptUse start: {0}, end: {1}, len: {2}'.format(cptUse[0],
                                                    #cptUse[-1], len(cptUse)))
    #print('lenCP: {0}'.format(len(cp)))
    
    # what does this stuff do I don't know...
    print('numBlocks: {0}, dataPts/Block: {1}'.format(numBlocks, len(x)/numBlocks))
    for idBlock in range(numBlocks):
        # set ii1, ii2 as indexes.  Fix edge case at end of blocks
        ii1 = int(cptUse[idBlock])
        if idBlock < (numBlocks - 1):
            ii2 = int(cptUse[idBlock + 1] - 1)
        else:
            ii2 = int(numPts)
                
        subset = y[ii1:ii2]
        weight = cellData[ii1:ii2]
        if ii1 == ii2:
            subset = y[ii1]
            weight = cellData[ii1]
        rateVec[idBlock] = np.dot(weight, subset) / np.sum(weight)

        if np.sum(weight) == 0:
            raise ValueError('error, divide by zero at index: {0}'.format(idBlock))
            print('-------ii1: {0}, ii2: {1}'.format(ii1, ii2))
            

    # Simple hill climbing for merging blocks
    cpUse = np.concatenate(([1], cp, [len(y)]))
    cp = cpUse

    numCP = len(cpUse) - 1
    idLeftVec = np.zeros(numCP)
    idRightVec = np.zeros(numCP)

    for i in range(numCP):
        idLeftVec[i]  = cpUse[i]
        idRightVec[i] = cpUse[i+1]

    # Find maxima defining watersheds, scan for 
    # highest neighbor of each block

    idMax = np.zeros(numBlocks)
    idMax = np.zeros(numBlocks)
    for j in range(numBlocks):
        jL = (j-1)*(j>0) + 0*(j<=0) # prevent out of bounds
        jR = (j+1)*(j<(numBlocks-1)) + (numBlocks-1)*(j>=(numBlocks-1))

        rateL = rateVec[jL]
        rateC = rateVec[j]
        rateR = rateVec[jR]
        rateList = [rateL, rateC, rateR]

        jMax = np.argmax(rateList) #get direction [0, 1, 2]
        idMax[j] = j + jMax - 1 # convert direction to delta

    idMax[ idMax > numBlocks] = numBlocks
    idMax[ idMax < 0] = 0

    # Implement hill climbing (HOP algorithm)
    
    hopIndex = np.array(range(numBlocks)) # init: all blocks point to self
    hopIndex = hopIndex.astype(int)  # cast all as int
    ctr = 0
    # point each block to its max block
    while ctr <= len(x):
        newIndex = idMax[hopIndex]  # Point each to highest neighbor

        if np.array_equal(newIndex, hopIndex):
            break
        else:
            hopIndex = newIndex.astype(int)
            
        ctr += 1
        if ctr == len(x):
            print('Hill climbing did not converge...?')

    idMax = np.unique(hopIndex)
    numMax = len(idMax)

    # Convert to simple list of block boundaries
    boundaries = [0]
    for k in range(numMax):
        currVec = np.where(hopIndex == idMax[k])[0]

        rightDatum = idRightVec[currVec[-1]] - 1 # stupid leftover matlab index
        boundaries.append(rightDatum)
    
    return np.array(boundaries)

