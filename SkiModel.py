#code for the ski model function


#numerical solutions
import numpy as np
from scipy.integrate import solve_bvp, simpson
import matplotlib.pyplot as plt

#def EI_func(s): return 200/(s**2 + 0.5) - 255

def EI_func(s, widths, skey, t_ski = .005):
    t = t_ski #m, thickness of ski, fixed
    E = 410*10**9 #Pa, youngs modulus of wood (to approximate values)
    width_vals = np.interp(s, skey, widths)

    return E*width_vals*np.pow(t, 3)/12.0

#returns the depth into the snow assuming the weight applied at a given angle
#this gives a contact area upon which snow forces act
def depth_into_snow(Fg, alpha, w, l_ski, t_ski, E_snow = 0.75):
    K_snow = E_snow*l_ski*t_ski*10**9
    w_depth = Fg/K_snow/np.sin(alpha)
    w_depth = np.minimum(w_depth, w)

    return w_depth

#old function for snow forces, kinda sketchy
def F_ice(h, w): 
    
    beta = np.diff(h)
    beta = np.append(beta, beta[-1])
    #beta = np.concatenate(beta[0], beta)
    print(beta.shape)
    beta = 3.14159/4
    return (16120/np.pow(np.sin(beta), 3.77) - 15510) * (39.37*np.maximum(h,0))**(-0.377*beta +.7)

#the main function for calculating snow "cutting" forces
def F_snow(h,w, s, vel, mu = 0.1, u_snow = 0.02): 

    #calculate the angle of the ski against incoming snow (like a machining rake angle)
    alphas = -(90 - np.arctan(np.gradient(h,s)))

    #the friction angle
    beta = np.arctan(mu)

    #the shear angle
    shear_angle = np.pi/4+alphas/2-beta/2

    #the specific energy of the ice
    ut_ice = 60*10^9 #in 

    #approximate Fc as ut*MRR/v_
    Fc = ut_ice*np.gradient(s)*w*vel

    #Get the trust force with "rake angle" and the friction angle
    Ft = Fc*np.tan(beta-alphas)

    #now get the force normal to the ski and the parallel friction force
    Ff = Fc*np.sin(alphas) + Ft*np.cos(alphas) #parallel to ski
    Fn = Fc*np.cos(alphas) - Ft*np.sin(alphas) #normal to ski

    return Ft, Ff, Fc, Fn #return all the force in case we need them later

    
def solve_beam(L, EI_func, Fs, widths, s0, t_ski):
    s = np.linspace(-L/2, L/2, 200)

    def ode(s, y):
        h, h1, h2, h3 = y
        #print(Fs(s))
        EI_vals = EI_func(s, widths, s0, t_ski)
        return np.vstack((h1, h2, h3, Fs(s)/EI_vals))

    def bc(ya, yb): return np.array([ya[0], ya[2], yb[0], yb[2]])

    y_init = np.zeros((4, s.size))
    sol = solve_bvp(ode, bc, s, y_init, tol=1e-4)
    return s, sol.sol(s)[0]

def get_widths(s, L, w, w_max, r_sc, l_sc):
    widths = np.zeros_like(s)

    for i in range(0,len(widths)):
        if np.abs(s[i]) > l_sc/2:
            widths[i] = w_max
        else:
            widths[i] = w + r_sc - np.sqrt(r_sc**2-np.abs(s[i])**2)

    return widths


def get_tranform_matrix(ax = 0.001, ay = 3.14159/4, az = .05, er = 0.0, et = 0.0):

    R1 = np.matrix([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    R2 = np.matrix([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    R3 = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])

    R = np.dot(np.dot(R1,R2), R3) #need to find a new class for matrices

    P1 = np.array([[1, 0, 0], [0, np.cos(et), -np.sin(et)], [0, np.sin(et), np.cos(et)]])
    P2 = np.array([[np.cos(er), 0, np.sin(er)], [0, 1, 0], [-np.sin(er), 0, np.cos(er)]])

    P = np.dot(P1,P2)
    RP = np.dot(R,P)
    #print(RP)
    n_fs = np.dot(RP, np.array([[0],[0],[1]])) #ignoring P matrices for now

    return n_fs


#could be adjusted to align better with paper, but for now, use some hardcoded assumptions
def solveFT_and_angles(Fs,s, ax = 0.001, ay = 3.14159/4, az = .05):
    Fs_sum = simpson(s*Fs, s) #get scalar value of force, and assume its pointing in the same dir the whole time (small deformation assumption)

    #set up the transform equations
    n_fs = get_tranform_matrix(ax, ay, az)
    #print(n_fs)

    Fs_dir = Fs_sum*n_fs

    Ft = Fs_dir[1] #assume that thrust balances the force in theta dir/y direction

    #now calculate Ft again, but find the change in it due to slight perterbation et
    n_fs2 = get_tranform_matrix(ax, ay, az, et = .001)
    #print(n_fs2)
    Fs_dir2 = Fs_sum*n_fs2

    Ft_del = Fs_dir2[1]

    diff_Ft = Ft_del-Ft

    return Ft, .001, diff_Ft
    

def plot_ski(s, widths):
    #print the widths for reference
    plt.figure()
    ax = plt.subplot()
    ax.plot(s,widths/2)
    plt.xlabel("length (m)")
    plt.ylabel("width (m)")
    ax.set_aspect(10)


#takes in a list of ski parameters in optimize_vector that builds a simple ski
#optimize_vector expected values: 0 - L, length of ski
                                # 1 - w, waist width, m
                                # 2 - w_max, max width, m
                                # 3 - r_sc, sidecut radius, m
                                # 4 - l_sc, sidecut length, m
                                # 5 - t_ski, thickness of ski (uniform), m
def ski_turn_iterative(optimize_vector,W_person=70, l_person=1.0, Kf=0.02,
                       max_iter=30, relax=0.4, tol=1e-3, az = 0, ax = 0):
    #ski parameters
    L = optimize_vector[0]
    w = optimize_vector[1]
    w_max = optimize_vector[2]
    r_sc = optimize_vector[3]
    l_sc = optimize_vector[4]
    t_ski = optimize_vector[5]
    #static parameters
    g = 9.81 #m/s, gravity
    R_curve = 10 #m, radius of curving turn
    theta_dot = 1 #rad/s, rate of going around the carving turn
    Fg = g*W_person #gravity force acting on person
    alphay = 3.14159/4 #angle at which person is skiing
    

    s = np.linspace(-L/2, L/2, 200)
    s0 = s.copy()
    h = .001*s**2  # initial flat ski
    widths = get_widths(s, L, w, w_max, r_sc, l_sc)
    EI_vals = EI_func(s, widths, s0, t_ski)

    #plot_ski(s,widths)

    for k in range(max_iter):
        Fs_vals = F_snow(h, depth_into_snow(Fg, alphay, widths, L, t_ski), s, R_curve*theta_dot, Kf)
        Fs_vals = Fs_vals[0]
        Fs = lambda s_interp: np.interp(s_interp, s, Fs_vals)
        
        s_new, h_new = solve_beam(L, EI_func, Fs, widths, s0, t_ski)
        h_new = np.interp(s, s_new, h_new)

        # relaxation update
        h_next = (1-relax)*h + relax*h_new
        err = np.linalg.norm(h_next - h)/max(1e-12, np.linalg.norm(h_next))
        h = h_next
        if err < tol:
            print(f"Converged in {k+1} iterations (err={err:.2e})")
            break

    Fs_final = F_snow(h, widths, s, R_curve*theta_dot, Kf)
    Fs_final = Fs_final[0]

    #add code to solve for angles here to get Ft and sensitivity
    Ft, et, diff_FT = solveFT_and_angles(Fs_final,s)
    efficiency = Ft / (W_person*g)
    sensitivity = (diff_FT-Ft)*l_person/et/(W_person*g*l_person)

    return s, h, Fs_final, Ft, efficiency, sensitivity



if __name__ == "__main__":

    #for testing different function outputs

    #Region
    
    L = 2.0 #m, length of ski
    w = 0.099 #m, waist width
    w_max = .125 #m, max width
    r_sc = 17.3 #m, sidecut radius
    l_sc = 1.31 #length of sidecut
    t_ski = .005 #thickness of ski


    #test the beam bending
    # s = np.linspace(-L/2, L/2, 2000)
    # F = np.ones_like(s)*100
    # widths = get_widths(s, L, w, w_max, r_sc, l_sc)
    # Fs = lambda s_interp: np.interp(s_interp, s, F)

    # s_0, h = solve_beam(L, EI_func, Fs, widths,s)

    # plt.figure()
    # plt.plot(s_0,h)
    # plt.xlabel("s")
    # plt.ylabel("h(s)")
    # plt.show()

    # plt.figure()
    # print(F[0])

    #trial run of full code
    input_vector = [L,w,w_max, r_sc, l_sc, t_ski] #input optimize vector in this form

    #visualize the ski 
    s_vis = np.linspace(-2.0/2, 2.0/2, 200)
    print(s_vis[0])
    widths = get_widths(s_vis, L, w, w_max, r_sc, l_sc)
    plot_ski(s_vis,widths)


    #get the efficiency & sensitivity (it's a little sketch but at least it gives different numbers)
    s, h, Fs, Ft, eff, sens = ski_turn_iterative(input_vector, tol = 1e-6) #can set weight and person length too if needed, current defaults: W_person=70, l_person=1.0
    #print(f"Ft = {Ft:.2f} N,  Efficiency = {eff:.4f},  Sensitivity = {sens:.4e}")
    print("efficiency", eff)
    print("sensitivity", sens)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(s, h); plt.ylabel("Deflection h(s) [m]")
    plt.subplot(2,1,2)
    plt.plot(s, Fs); plt.ylabel("Snow force F_s(s) [N/m]")
    plt.xlabel("s (m)")
    plt.tight_layout(); plt.show()


    

   