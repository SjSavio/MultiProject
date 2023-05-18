import numpy as np

def get_prandtl_meyer(mach, gamma=1.4):
    '''Evaluate Prandtl-Meyer function at given Mach number.
    
    Defined as the angle from the flow direction where Mach = 1 through which the 
    flow turned isentropically reaches the specified Mach number.
    '''
    return (
        np.sqrt((gamma + 1) / (gamma - 1)) *
        np.arctan(np.sqrt((gamma - 1)*(mach**2 - 1)/(gamma + 1))) -
        np.arctan(np.sqrt(mach**2 - 1))
        )


def solve_prandtl_meyer(mach, nu, gamma=1.4):
    '''Solve for unknown Mach number, given Prandtl-Meyer function (in radians).'''
    return (nu - get_prandtl_meyer(mach, gamma))


def get_mach_angle(mach):
    '''Returns Mach angle'''
    return np.arcsin(1.0 / mach)