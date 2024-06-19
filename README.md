Curba de Rendimientos

Code Walk through:
1) Get data from excel using open source "pandas" library, by finding the name (rendimiento and plazo dias)- note, the index is set to this amount of data, so if more data is added, the index needs to be updated
           .create 4 categories:
                 rendimiento_quetzales
                 plazodias_quetzales
                 repeat for dolares
3) Sort the data from smallest to largest and drop potential null values, add data to respective lists
4) using open source matplotlib.pyplot library, plot the datapoints that we have (simple linear plot)
      .using zip, match x values to y values as fail proof
      .plot

** up until this point, graph is the same as current system

5) linear approixmations are used to simulate data by creating a polynomial equation to a defined degree, for ex degree 5 would create ax^5 + bx^4 + cx^3 ... + dx + e (where a...z are constants)
        .as a result, because it is a polynomial, it is more curvy (measured by "curvature")
        .however, when there are not many data points, the approximation will not be tight with the data, (there will be too much curvature, and it will not look accurate)
        . the "Chebyshev" approximation is one type of linear approximation, there are also other types like a Cubic Spline
6) question: How to have a lot of data to have normal curvature, and avoid too much curvature with current data?
7) solution: using linear graph, plot 100 new data points along these lines.  (the linear graph is just y = mx + b between points, use these "equations" to find new points along the line)
8) use open source "numpy" library to do this automatically
       -> quetzal_interp_x = np.linspace(min(plazo_dias_data_q_sorted), max(plazo_dias_data_q_sorted), 420) #420 new points
9) with updated data, use numpy again to plot a Chebyshev (open source method included in numpy)
   -
                  # Define the number of Chebyshev coefficients to use for approximation
                  degree = 5
                  
                  # Fit Chebyshev polynomial to the Quetzal data
                  quetzal_cheb_fit = np.polynomial.Chebyshev.fit(quetzal_interp_x, quetzal_interp_y, degree)
                  
                  # Fit Chebyshev polynomial to the Dollar data
                  dollar_cheb_fit = np.polynomial.Chebyshev.fit(dollar_interp_x, dollar_interp_y, degree)
                  
                  # Generate the Chebyshev approximation curve for Quetzal
                  quetzal_cheb_curve = quetzal_cheb_fit(quetzal_interp_x) 
                  
                  # Generate the Chebyshev approximation curve for Dollar
                  dollar_cheb_curve = dollar_cheb_fit(dollar_interp_x)

10) this generates the equation, next plot it the same way as the linear equation
   . There are 2 ways we can evaluate the curve, mean squared error(MSE) which is the difference between the curve and the data points, and the curvature
   . the NSS has lower MSE than then the Cheb, but also has lower curvature than the Cheb. the combined fit, which combines NSS and linear, is probably the least effective one since it just looks pretty linear
   . would recommend to use an algorithim that alternates between NSS and Cheb
       . when there is enough data, and the curvature of the NSS is higher than "x" desired amount, use NSS
       . when there is not enough data, or the curvature of the NSS is very low, use Cheb
       . Or just use Cheb

*nelson svennson is similar to Chebyshev, but was specifcally made for bond yields, however that is assuming there is a lot of data that are not super consistant (not very linear). Therefore the NSS can ussualy look very curvy when the data allows it. But when there isnot much data, or the data is very linear, the NSS is also very linear
