import numpy as np
from helperFuntions import *
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

gamma = 1.4 #specific heat

wall_angle = 6 * np.pi / 180 #angle of divergent section wall in radians

M_initial = 2.0 #speed of fluid is 2x speed of sound before entering divergent section

#will be looking at a grid of points and finding properties at each point
num_initial = 4 #number of points we will be initially examining

num_columns = 5 

total_points = (num_initial * num_columns + (num_initial-1) * (num_columns-1) )


thetas = np.zeros(total_points) #represents angle of flow at each point
machs = np.zeros(total_points) #speed of flow at each point
nus = np.zeros(total_points) #used along with theta value to find new points
ys = np.zeros(total_points) #y value of each point
xs = np.zeros(total_points) #x value of each point


#kplus = theta-mu
#kminus = theta+mu
#these are used to represent each characteristic line, all
#points on each characteristic line share the same k value
kplus = np.zeros(num_initial - 1)
kminus = np.zeros(num_initial - 1)

#the initial points will lay on an arc and will all have the same Mach number
#the complete arc will be the angle of the wall angle

#diff in theta between all the initial points
dTheta = wall_angle/(num_initial-1)

#setting the thetas for the inital points that
#lie on the arc
thetas[:num_initial] = [
    wall_angle - dTheta * i for i in range(num_initial)
    ]
#speed of all initial points on arc
#is set to initial speed
machs[0:num_initial] = M_initial

#assings nu value to initial points
nus[:num_initial] = get_prandtl_meyer(machs[:num_initial], gamma)

#x-value for first point is on the wall and is situated so that 
#y value is 1 unit high
xs[0] = 1/np.tan(wall_angle)
ys[0] = 1

#will assign x and y values to other initial points
radius = np.sqrt(xs[0]**2 + ys[0]**2)
xs[1:num_initial] = radius * np.cos(thetas[1:num_initial])
ys[1:num_initial] = radius * np.sin(thetas[1:num_initial])

#calculate k values/characteristics
#for all initial points in the arc
#except last point
for i in range(num_initial - 1):
    kminus[i] = thetas[i] + get_prandtl_meyer(machs[i], gamma)
    #we put i+1 here because it will simplify the process
    #of finding new points in the next few steps
    kplus[i] = thetas[i+1] - get_prandtl_meyer(machs[i+1], gamma)

#now we will move on to the next coulumn of points

for i in range(num_initial-1):
    #we get new angle of flow by adding the characteristics of the 
    #two points that are in the column from before
    #essentially we are finding a new point that meets the 
    #characteristics lines of the previous two points

    #the i+1 from line 67 provides an advantage in this step
    thetas[i+num_initial] = 0.5 * (kminus[i] + kplus[i])
    nus[i+num_initial] = 0.5 * (kminus[i] - kplus[i])

    #now since we have theta and nu for the new point
    #we will calculate the the Mach/Speed of the flow
    #at this new point, note we don't know where this 
    #point actually is yet!
    #rooot scalar function will find the correct M 
    #value that is closest to giving 
    #the desired nus value
    root = root_scalar( solve_prandtl_meyer, x0=2.0, x1=3.0, args=(nus[num_initial + i], gamma))
    machs[num_initial + i] = root.root

    mu = get_mach_angle(machs[num_initial + i])
    # dydx_minus is the slope of the characteristic line
    # from kminus, we will use this info along with
    # dydx_plus to find the value of the coordinate
    # of this new point
    dydx_minus = np.tan(0.5*(thetas[i] + thetas[num_initial + i] - get_mach_angle(machs[i]) - mu))
    dydx_plus = np.tan(0.5*(thetas[i + 1] + thetas[num_initial + i] + get_mach_angle(machs[i + 1]) + mu ))
    #new x and y value of this new point
    xs[num_initial + i] = (ys[i+1] - ys[i] - xs[i+1] * dydx_plus + xs[i] * dydx_minus) / (dydx_minus - dydx_plus)
    ys[num_initial + i] = (ys[i] + (xs[num_initial + i] - xs[i])*dydx_minus)


#now we do the rest of the points
for icol in range(1,num_columns):
    #represents the index of new points we're focusing on
    i_start = icol * (2*num_initial-1)
    for i in range(num_initial):
        # we will first determine the characteristics of 
        # the point on the wall/nozzle
        if i == 0:

            #since point is on nozzle boundary
            # the fluid flow will have a velocity
            # that shares the same angle as the
            # wall angle

            # nu can be calculated by using  the
            # characteristic from the point
            # of the previous column
            thetas[i_start + i] = wall_angle
            nus[i_start + i] = wall_angle + nus[i_start+i-num_initial+1] -thetas[i_start+i-num_initial+1]
            
            # since we have nu we can calculate the speed 
            # of the fluid at this boundary point
            root = root_scalar( solve_prandtl_meyer, x0=2.0, x1=3.0, args=(nus[i_start + i], gamma))

            machs[i_start+i] = root.root
            mu = get_mach_angle(machs[i_start + i])

            # now we will calculate the slope of the 
            # characteristic line that intersects this 
            # new boundary point
            dydx_plus = np.tan(0.5*(
                thetas[i_start + i - num_initial + 1] + 
                thetas[i_start + i] + 
                get_mach_angle(machs[i_start + i - num_initial + 1]) + mu
                ))


            # using some geometry we will calculate the
            # x and y value of this point on the nozzle
            x1 = xs[i_start + i - num_initial + 1]
            y1 = ys[i_start + i - num_initial + 1]
            
            x0 = x1
            y0 = ys[0] + (x0 - xs[0]) * np.tan(wall_angle)

            xs[i_start + i] = (
                y0 - y1 - x0*np.tan(wall_angle) + x1*dydx_plus
                ) / (dydx_plus - np.tan(wall_angle))
            ys[i_start + i] = (
                y0 + (xs[i_start + i] - x0) * np.tan(wall_angle)
                )

        # point on line of symetry
        elif i == num_initial - 1:
            # angle of flow will be 0 
            # since its moving horizontally
            # nu will be calculated by basically
            # looking at the kminus value of a point
            # that is located on the column from before
            thetas[i_start + i] = 0
            nus[i_start+i] = (nus[i_start + i - num_initial] + thetas[i_start + i - num_initial])

            root = root_scalar(solve_prandtl_meyer, x0=2.0, x1=3.0, args=(nus[i_start + i], gamma))
            machs[i_start + i] = root.root
            mu = get_mach_angle(machs[i_start + i])

            # calculate dydx to get slope
            # so that we can find x coordinate of
            # new point

            dydx_minus = np.tan(0.5*(
                thetas[i_start + i - num_initial] +
                thetas[i_start + i] -
                get_mach_angle(machs[i_start + i - num_initial]) - mu
                ))

           
            # y cooridnate 0 because this point
            # lays on the line of symmetry
            x1 = xs[i_start + i - num_initial]
            y1 = ys[i_start + i - num_initial]
            
            xs[i_start + i] = (
                x1 - y1 / dydx_minus
                )
            ys[i_start + i] = 0           


        # now we will consider all other points
        else:
            # all of this is basically the same as
            # the code lines 71-99
            thetas[i_start + i] = 0.5 * (kminus[i-1] + kplus[i])
            nus[i_start + i] = 0.5 * (kminus[i-1] - kplus[i])
            root = root_scalar(solve_prandtl_meyer, x0=2.0, x1=3.0,  args=(nus[i_start + i], gamma))
            machs[i_start + i] = root.root
            mu = get_mach_angle(machs[i_start + i])
            
            dydx_minus = np.tan(0.5*(thetas[i_start + i - num_initial] + thetas[i_start + i] - get_mach_angle(machs[i_start + i - num_initial]) - mu))
            dydx_plus = np.tan(0.5*(thetas[i_start + i - num_initial + 1] + thetas[i_start + i] + get_mach_angle(machs[i_start + i - num_initial + 1]) + mu))

            xs[i_start + i] = ( ys[i_start + i - num_initial + 1] - ys[i_start + i - num_initial] - xs[i_start + i - num_initial + 1]*dydx_plus + xs[i_start + i - num_initial]*dydx_minus) / (dydx_minus - dydx_plus)
            ys[i_start + i] = (ys[i_start + i - num_initial] + (xs[i_start + i] - xs[i_start + i - num_initial])*dydx_minus)

    # now we will find points in the next column
    # we are shifting 
    # we dont need to have new k values of the characteristic lines
    # for the new points since they will have the same 
    # k value as the characteristic lines formed by the two previous points 
    kminus[1:] = kminus[:-1]
    kminus[0] = thetas[i_start] + nus[i_start]
    kplus[:-1] = kplus[1:]        
    kplus[-1] = (thetas[i_start + num_initial - 1] - nus[i_start + num_initial - 1])

    #we are starting another column
    i_start += num_initial
    #this column will have num_initial-1 points
    if icol < num_columns - 1:
        for i in range(num_initial-1):
            #again this is all basically the same as lines 71-99
            thetas[i_start + i] = 0.5 * (kminus[i] + kplus[i])
            nus[i_start + i] = 0.5 * (kminus[i] - kplus[i])
            root = root_scalar(solve_prandtl_meyer, x0=2.0, x1=3.0, args=(nus[i_start + i], gamma))
            machs[i_start + i] = root.root
            mu = get_mach_angle(machs[i_start + i])
                
            dydx_minus = np.tan(0.5*(thetas[i_start + i - num_initial] + thetas[num_initial + i] - get_mach_angle(machs[i_start + i - num_initial]) - mu))
            dydx_plus = np.tan(0.5*(thetas[i_start + i - num_initial + 1] + thetas[i_start + i] + get_mach_angle(machs[i_start + i - num_initial + 1]) + mu))

            xs[i_start + i] = (ys[i_start + i - num_initial + 1] - ys[i_start + i - num_initial] - xs[i_start + i - num_initial + 1]*dydx_plus + xs[i_start + i - num_initial]*dydx_minus ) / (dydx_minus - dydx_plus)
            ys[i_start + i] = ( ys[i_start + i - num_initial] +(xs[i_start+i] - xs[i_start + i - num_initial])*dydx_minus )


#plotting the data
plt.plot(xs, ys, 'o')

plt.plot(
    [xs[num_initial-1], xs[-1]], 
    [ys[num_initial-1], ys[-1]], '-.k'
    )
plt.plot(
    [xs[0], xs[-num_initial]], [ys[0], ys[-num_initial]], '-b', 
    )


for idx, (x,y) in enumerate(zip(xs, ys)):
    plt.text(
        x, y + 0.02, f'{idx: d}', 
        verticalalignment='bottom', horizontalalignment='center'
        )
    
for i in range(total_points):
    plt.arrow(xs[i],ys[i], machs[i]*0.1*np.cos(thetas[i]), machs[i]*0.1*np.sin(thetas[i]), head_width=0.02, head_length = 0.05)


plt.ylim([-0.05, np.max(ys)+0.1])

plt.tight_layout()

plt.show()

#printing speed and angle of each point
print('id   Mach     angle (°)')
for i, (mach, theta) in enumerate(zip(machs, thetas)):
    print(f'{i: d}  {mach: .4f}  {theta*180/np.pi: .3f}')


