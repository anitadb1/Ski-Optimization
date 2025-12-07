#alternative Ski model code with less iterative solving
#12/4/2025

#code for the ski model function


#numerical solutions
import numpy as np
from scipy.integrate import solve_bvp, simpson
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

#def EI_func(s): return 200/(s**2 + 0.5) - 255

def EI_func(s, widths, skey, t_ski = .005):
    t = t_ski #m, thickness of ski, fixed
    E = 10*10**9 #Pa, youngs modulus of wood (to approximate values)
    width_vals = np.interp(s, skey, widths)

    return E*width_vals*np.pow(t, 3)/12.0

#returns the depth into the snow assuming the weight applied at a given angle
#this gives a contact area upon which snow forces act
def depth_into_snow(Fg, alpha, w, l_ski, t_ski, E_snow = 0.75):
    K_snow = E_snow*l_ski*t_ski*10**9
    w_depth = Fg/K_snow/np.sin(alpha)
    w_depth = np.minimum(w_depth, w)

    return w_depth

#the main function for calculating snow "cutting" forces
def F_snow(h,w, s, vel, mu = 0.1, u_snow = 0.02): 

    #calculate the angle of the ski against incoming snow (like a machining rake angle)
    alphas = -(90 - np.arctan(np.gradient(h,s)))

    #the friction angle
    beta = np.arctan(mu)

    #the shear angle
    shear_angle = np.pi/4+alphas/2-beta/2

    #the specific energy of the ice
    ut_ice = 60*10^6 #in 

    #approximate Fc as ut*MRR/v_
    Fc = ut_ice*np.gradient(s)*w/4

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

def get_widths_smooth(s, L, w, w_max, r_sc, l_sc, smooth=1e-4):
    # --- original raw width generation ---
    widths = np.zeros_like(s)

    for i in range(len(widths)):
        if np.abs(s[i]) > l_sc/2:
            widths[i] = w_max
        else:
            widths[i] = w + r_sc - np.sqrt(r_sc**2 - np.abs(s[i])**2)

    # --- smoothing step ---
    # create a spline over the original (s, widths) data
    spline = UnivariateSpline(s, widths, s=smooth)

    # evaluate smoothed curve
    widths_smooth = spline(s)

    return widths_smooth



    

def plot_ski(s, widths):
    #print the widths for reference
    set_text(10,1)
    plt.figure()
    ax = plt.subplot()

    top = widths / 2
    bottom = -widths / 2

    # Plot the ski edges
    ax.plot(s, top, color='red', linewidth=2, alpha=0.25)
    ax.plot(s, bottom, color='red', linewidth=2, alpha=0.25)

    # Fill between edges (transparent color)
    ax.fill_between(
        s,
        top,
        bottom,
        color='#cc2143',   # choose your color here
        alpha=0.3          # transparency (0=transparent, 1=solid)
    )

    # Labels and aspect ratio
    ax.set_xlabel("length (m)")
    ax.set_ylabel("width (m)")

    # Aspect ratio: (this keeps things more visually realistic)
    ax.set_aspect(1)
    

    plt.show()
    
def set_text(w, h):
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['figure.figsize'] = (w, h)
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False

def plot_ski_from_vector(optimize_vector):
    L = optimize_vector[0]
    w = optimize_vector[1]
    w_max = optimize_vector[2]
    r_sc = optimize_vector[3]
    l_sc = optimize_vector[4]
    t_ski = optimize_vector[5]

    s = np.linspace(-L/2, L/2, 500)
    widths = get_widths_smooth(s, L, w, w_max, r_sc, l_sc)

    plot_ski(s,widths)
    #plt.title("L = " + str(L) + " w = " + str(w) + " w_sc = " + str(w_max) + " r_sc = " + str(r_sc) + " l_sc = " + str(l_sc) + " t = " + str(t_ski))



#takes in a list of ski parameters in optimize_vector that builds a simple ski
#optimize_vector expected values: 0 - L, length of ski
                                # 1 - w, waist width, m
                                # 2 - w_max, max width, m
                                # 3 - r_sc, sidecut radius, m
                                # 4 - l_sc, sidecut length, m
                                # 5 - t_ski, thickness of ski (uniform), m
#other parameters: 
    #v_ski - speed of skier (assuming that they have picked up speed going down the hill)
    #l_1person - distance from knee to com
    #l_2person - distance from boot knee
def ski_turn_iterative(optimize_vector,W_person=70, v_ski = 9, l_1person=.2,l_2person = 0.5, Kf=0.02):
    
    #ski parameters
    L = optimize_vector[0]
    w = optimize_vector[1]
    w_max = optimize_vector[2]
    r_sc = optimize_vector[3]
    l_sc = optimize_vector[4]
    t_ski = optimize_vector[5]

    #static parameters
    g = 9.81 #m/s, gravity
    #R_curve = 10 #m, radius of curving turn
    #theta_dot = 1 #rad/s, rate of going around the carving turn
    Fg = g*W_person #gravity force acting on person
    
    #alphay = 3.14159/4 #angle at which person is skiing
    #print(v_ski**2/g/r_sc)
    #value for tilt angle phi
    phi = np.arcsin(v_ski**2/g/r_sc)

    #value for the turn radius
    R_con = np.sqrt((g*r_sc)**2 - v_ski**2)/g

    #Now get the knee torques
    Fc = W_person*v_ski**2/R_con
    l1 = l_1person
    l2 = l_2person
    tau_knee = Fg*np.sin(phi)*(l1+l2) - Fc*np.cos(phi)*(l1 + l2)

    #get reaction force
    F_l = W_person*v_ski**2/R_con*np.sin(phi) + Fg*np.cos(phi)*np.sin(phi)
    #print(F_l)
    s = np.linspace(-L/2, L/2, 200)
    s0 = s.copy()
    #h = .001*s**2  # initial flat ski
    widths = get_widths(s, L, w, w_max, r_sc, l_sc)
    EI_vals = EI_func(s, widths, s0, t_ski)

    #make the load a vector
    F_load_for_solver = np.zeros_like(s)
    #print(np.round(len(F_load_for_solver)/2))
    F_load_for_solver[int(np.round(len(F_load_for_solver)/2))] = -F_l

    #solve for deflection
    Fs = lambda s_interp: np.interp(s_interp, s, F_load_for_solver)
    s_new, h_new = solve_beam(L, EI_func, Fs, widths, s0, t_ski)

    #solve for the friction force
    F_t, F_f, F_idk, F_idk2 = F_snow(h_new, w, s_new, v_ski, Kf)

    F_f2 = simpson(s*F_f, s)

    #parameters to return
    R_con
    tau_knee
    Efficiency = np.log10(-1/(F_f2/Fg))



    return s_new, h_new, F_f, R_con, tau_knee, Efficiency



if __name__ == "__main__":

    #for testing different function outputs

    #Region
    
    L = 1.048 #m, length of ski
    w = 0.0972 #m, waist width
    w_max = .1301 #m, max width
    r_sc = 10.3 #m, sidecut radius
    l_sc = 1.28 #length of sidecut
    t_ski = .00991 #thickness of ski


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
    #print(s_vis[0])
    widths = get_widths_smooth(s_vis, L, w, w_max, r_sc, l_sc)
    #plot_ski(s_vis,widths)
    plot_ski_from_vector([L,w, w_max, r_sc, l_sc, t_ski])

    #get the efficiency & sensitivity (it's a little sketch but at least it gives different numbers)
    s,h, Fs, R_con, tau_knee, eff = ski_turn_iterative(input_vector, v_ski = 10) #can set weight and person length too if needed, current defaults: W_person=70, l_person=1.0
    #print(f"Ft = {Ft:.2f} N,  Efficiency = {eff:.4f},  Sensitivity = {sens:.4e}")
    print("efficiency", eff)
    print("R_con", R_con)
    print("Knee Torque", tau_knee)
    #print(Fs)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(s, h); plt.ylabel("Deflection h(s) [m]")
    plt.subplot(2,1,2)
    plt.plot(s, Fs); plt.ylabel("Snow force F_s(s) [N/m]")
    plt.xlabel("s (m)")
    plt.tight_layout(); plt.show()


    

   

