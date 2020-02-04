def bkgdSub(self, **kwarg):
        '''
        Perform modified chebyshev polynomial subtraction and penalize with cost fn
        Offset data to be positive
    
        Adapted from Fang Ren segmentation program

        '''
        
        # DONT Truncate data if processing from tif
        trimDataX = self.data[0, 50:-50]
        trimDataY = self.data[1, 50:-50]

        def downsamp(size, x, y):
            '''
            downsample data based on size parameter.  Return downsampled data
            '''
            yNew = y[::size] 
            xNew = x[::size] 
            
            return xNew, yNew
        
        def chebFunc(x, *params):
            '''
            Modified chebyshev function with 1/x 
            '''
            params = params[0]
            y = chebval(x, params[:4])
            E = params[4]
            y = y + E/x
            return y


        def objFunc(*params):
            '''
            implement cost function for modified chevbyshev function
            varies cost to emphasize low Q region
            Scoping pulls trimmed x, y data from surrounding scope

            return: cost of fit 
            '''
            
            params = params[0]
            J = 0
            fit = chebval(X, params[:4])
            E = params[4]
            fit = fit + E / X

            for i in range(len(Y)):
                if X[i] < 1:    # Treat low Q equally
                    J = J + (Y[i] - fit[i])**4
                else:
                    if Y[i] < fit[i]:
                        J = J + (Y[i] - fit[i])**4
                    if Y[i] >= fit[i]:
                        J = J + (Y[i] - fit[i])**2
            return J
        
        # Create a sparse data set for fitting
        X, Y = downsamp(50, trimDataX, trimDataY)
       
        # Remove points above median + 10% of range
        medianY = np.median(trimDataY)
        rangeY = np.max(trimDataY) - np.min(trimDataY)
        X = X[Y <= (medianY + 0.1*rangeY)]
        Y = Y[Y <= (medianY + 0.1*rangeY)]
        
        # Re-append end values to try and fix boundary conditions
        X = np.append(np.append(trimDataX[0:30:5], X), trimDataX[-30:-1:5])
        Y = np.append(np.append(trimDataY[0:30:5], Y), trimDataY[-30:-1:5])


        x0 = [1,1,1,1,1]
        #appears to converge quickly, take 10 iterations rather than 100
        result = basinhopping(objFunc, x0, niter=10) 
        bkgd_sparse = chebFunc(trimDataX, result.x)
        # create function that interpolates sparse bkgd
        f = interp1d(trimDataX, bkgd_sparse, kind='cubic', bounds_error=False)

        # expressed background values
        bkgd1 = f(trimDataX)
        subDataY = trimDataY - bkgd1

        # Dump any nan values in all data
        finalDataY = subDataY[~np.isnan(subDataY)]
        finalDataX = trimDataX[~np.isnan(subDataY)]
        bkgd = bkgd1[~np.isnan(subDataY)]

        # Offset to make all values > 0 
        if np.min(subDataY) < 0:
            finalDataY = finalDataY + np.absolute(np.min(finalDataY))
        
        # Save background subtracted data
        self.subData = np.array([finalDataX, finalDataY])
        self.bkgd = bkgd
        self.downData = np.array([X, Y])
        
        print '[[ Chevbyshev ]] background sub completed'