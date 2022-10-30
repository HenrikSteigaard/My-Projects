import sympy as sp
import numpy as np


def main():
   
    t1, t2, t3 = sp.symbols('theta_1 theta_2 theta_3')
    l1, l2, l3 = sp.symbols('L_1 L_2 L_3')
    s1, s2, s3 = sp.symbols('s_1 s_2 s_3')
    c1, c2, c3 = sp.symbols('c_1 c_2 c_3')
    
    print("\n** The Jacobian for the 3 DoF manipulator is given as follows below ** \n")
    
    jacobian = sp.Matrix([ [-s1*(l2*c2+l3*c2*c3), -c1*(l2*s2+l3*s2*s3), -c1*(l3*s2*s3)],
                          [c1*(l2*c2+l3*c2*c3), -s1*(l2*s2 + l3*s2*s3), -s1*(l3*s2*s3)],
                          [0, (l2*c2+l3*c2*c3), l3*c2*c3],
                          [0, s1, s1],
                          [0, -c1, -c1],
                          [1, 0, 0] ])
    
    pprint(jacobian)
    
    compute_energies(jacobian)


def compute_energies(jacobian):
    
    t1, t2, t3 = sp.symbols('theta_1 theta_2 theta_3')
    l1, l2, l3 = sp.symbols('L_1 L_2 L_3')
    s1, s2, s3 = sp.symbols('s_1 s_2 s_3')
    c1, c2, c3 = sp.symbols('c_1 c_2 c_3')
    
    print("\n Task 2A) -------------------------------------------------------------------@) \n")
    
    print("\n Important Notice! \n\n\
        ** To find the r_ci vectors from the origin in the base coordinate system\n \
        , we can utilize the complete Homogenous transformation matrix from mandatory\n \
        assigment 2. The way we do this, is by substituting L1, ..., Ln with the \n \
        joint length to the current mass center, while disregarding all values that follow \n \
        Li by substituing them with zeros. If we do this in the X, Y, Z (1_column x 3_rows), \n \
        we do get the vector from the origin in the base coordinate system, to the center of \n \
        mass link i. **")
    
    g = sp.symbols("g")
    
    # the three masses for each mass m_i
    masses = [0.3833, 0.2724, 0.1406]
    
    # the three r_ci vectors, which have been computed according to the method above
    r_ci = [sp.Matrix([0, 0, l1/2]), sp.Matrix([l2/2*c1*c2, l2/2*s1*c2, l2/2*s2+l1]),
            sp.Matrix([l3/2*c1*c2*c3 - l3/2*s2*s3*c1+l2*c1*c2, 
            l3/2*s1*c2*c3-l3/2*s1*s2*s3+l2*s1*c2,
            l3/2*s2*c3+l3/2*s3*c2+l2*s2+l1])]
    
    gravi = sp.Matrix([0,0, g])
    g_transposed = sp.transpose(gravi)
  
    potential_energies = 0        
    print("\n")
    for i in range(len(masses)):
        
        print("Mass ", i+1, " potential energy is given as: ")
        pprint(masses[i] * g_transposed @ r_ci[i])
        print("\n")
        potential_energies +=np.ravel(masses[i] * g_transposed @ r_ci[i]) 
    
    print("\n** The total potential energy P is given as:\n")
    pprint(potential_energies)
    print("\n")
    
    print("\n Task 2B) -------------------------------------------------------------------@) \n")
    
    theta = sp.symbols("theta")
    i1x, i1y, i1z = sp.symbols("I_1x I_1y I_1_z")
    i2x, i2y, i2z = sp.symbols("I_2x I_2y I_2z")
    i3x, i3y, i3z = sp.symbols("I_3x I_3y I_3z")
    
    R_Z_theta = sp.Matrix([[sp.cos(theta), -sp.sin(theta), 0], 
                           [sp.sin(theta), sp.cos(theta), 0],
                           [0, 0, 1]])
    
    R_M1 = sp.eye(3)
    R_M2 = R_M1 @ R_Z_theta.subs(theta, t2)
    R_M3 = R_M2 @ R_Z_theta.subs(theta, t1)
    
    R_M1_T = sp.transpose(R_M1)
    R_M2_T = sp.transpose(R_M2)
    R_M3_T = sp.transpose(R_M3)
    
    I_1 = sp.Matrix([[i1x, 0, 0],
                     [0, i1y, 0],
                     [0, 0, i1z]])
    
    I_2 = sp.Matrix([[i2x, 0, 0],
                     [0, i2y, 0],
                     [0, 0, i2z]])
    
    I_3 = sp.Matrix([[i3x, 0, 0],
                     [0, i3y, 0],
                     [0, 0, i3z]]) 
    
    inertia_1 = R_M1 @ I_1 @ R_M1_T
    inertia_2 = R_M2 @ I_2 @ R_M2_T
    inertia_3 = R_M3 @ I_3 @ R_M3_T
    
    t1dot, t2dot, t3dot = sp.symbols("thetahat_1, thetahat_2, thetahat_3")

    q_dot = sp.Matrix([t1dot, t2dot, t3dot])
    q_dot_T = sp.transpose(q_dot)
    
    J_v1 = jacobian[0:3, 0:3].subs(((c2, 0), (s2, 0), (c3, 0), (s3, 0), (l2, 0), (l3, 0)))
    J_v2 = jacobian[0:3, 0:3].subs(((c3, 0), (s3, 0), (l3, 0)))
    J_v3 = jacobian[0:3, 0:3]
    
    J_v1_T = sp.transpose(J_v1)
    J_v2_T = sp.transpose(J_v2)
    J_v3_T = sp.transpose(J_v3)
    
    J_ang1 = jacobian[3:, 0:3].subs(((c2, 0), (s2, 0), (c3, 0), (s3, 0), (l2, 0), (l3, 0)))
    J_ang2 = jacobian[3:, 0:3].subs(((c3, 0), (s3, 0), (l3, 0)))
    J_ang3 = jacobian[3:, 0:3]
    
    J_ang1_T = sp.transpose(J_ang1)
    J_ang2_T = sp.transpose(J_ang2)
    J_ang3_T = sp.transpose(J_ang3)
    
    D_q_1 = masses[0] * J_v1_T @ J_v1 + J_ang1_T @ inertia_1 @ J_ang1
    D_q_2 = masses[1] * J_v2_T @ J_v2 + J_ang2_T @ inertia_2 @ J_ang2
    D_q_3 = masses[2] * J_v3_T @ J_v3 + J_ang3_T @ inertia_3 @ J_ang3
   
    # The total Kinetic energy, according to equation ( 7.58 ) in the course book:
    K = 1/2 * q_dot_T @ ( D_q_1 + D_q_2 + D_q_3 ) @ q_dot
    
    print("\n** The total kinetic energy K is given as expression: \n")
    print("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#")
    K = K[0:1]
    K = K[0]
    c1_n, c2_n, c3_n = sp.cos(t1), sp.cos(t2), sp.cos(t3)
    s1_n, s2_n, s3_n = sp.sin(t1), sp.sin(t2), sp.sin(t3)
    K = K.subs(((c1,c1_n), (c2, c2_n), (c3, c3_n), (s1, s1_n), (s2, s2_n), (s3, s3_n)))
    pprint(K)
    print("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#")
    
    print("\n Task 2E) -------------------------------------------------------------------@) \n")
    P = potential_energies[0]
    
    #The Lagrangian equation
    L = K - P
    
    print("\n** The Lagrangian L, given as L = K - P, is given as : \n")
    # pprint(L)
 
    # step 1: Calculating the partial derivative with respect to each joint variable k, k = 1, 2, 3
    
    # diff joint 1
    diff_t1 = sp.diff(L, t1)
    
    # diff joint 2
    diff_t2 = sp.diff(L, t2)
    
    # diff joint 3
    diff_t3 = sp.diff(L, t3)
    
    # step 2: Calculating the partial derivative with respect to each "joint hat"
    
    # diff joint hat 1
    diff_t1_hat = sp.diff(L, t1dot)
    
    # diff joint hat 2
    diff_t2_hat = sp.diff(L, t2dot)
    
    # diff joint hat 3
    diff_t3_hat = sp.diff(L, t3dot)
    
    # step 3: finding the time derivative of each partial derivative with respect to each "joint hat"
    
    # to tackle this, we need to do some minor modifications so that our program can find
    # all the time variables we want to differniate with respect to, that is theta_1_hat, theta_2_hat
    # and theta_3_hat as our time variables.
    
    # our modifications:
    t = sp.symbols("t")
    t1_hat, t2_hat, t3_hat = sp.Function('thetahat_1'), sp.Function('thetahat_2'), sp.Function('thetahat_3')
    diff_t1_hat = diff_t1_hat.subs( ( (t1dot, t1_hat(t) ), ( t2dot, t2_hat(t) ), ( t3dot, t3_hat(t) )))
    diff_t2_hat = diff_t2_hat.subs( ( (t1dot, t1_hat(t) ), ( t2dot, t2_hat(t) ), ( t3dot, t3_hat(t) )))
    diff_t3_hat = diff_t3_hat.subs( ( (t1dot, t1_hat(t) ), ( t2dot, t2_hat(t) ), ( t3dot, t3_hat(t) )))
 
    # time derivative of partial derivative for joint 1
    time_diff_t1 = sp.diff(diff_t1_hat, t)
    
    # time derivative of partial derivative for joint 2
    time_diff_t2 = sp.diff(diff_t2_hat, t)
    
    # time derivative of partial derivative for joint 3
    time_diff_t3 = sp.diff(diff_t3_hat, t)
    
    # step 4: we now have all componets neccessary to compose the final product:
    # the Euler Lagrange Equations:
    
    # Euler-Lagrange for Tau_1:
    tau_1 = time_diff_t1 - diff_t1
    
    # Euler-Lagrange for Tau_2:
    tau_2 = time_diff_t2 - diff_t2
    
    # Euler-Lagrange for Tau_3:
    tau_3 = time_diff_t3 - diff_t3
    
    # the completa Euler-Lagrange Tau is composed of all the different torques denoted by
    # tau_1, ..., tau_n as a 3x1 matrix (multiple ways of expressing the complete Tau)
    
    # and there you have it, the complete Tau:
    tau = sp.Matrix([tau_1,tau_2,tau_3])
    
    # Thoughts: The general implementation for tasks 2A, 2B and 2E could have been better:
    # for instance, I would have done a more proper and organized incapsulation by parting
    # components of the tasks into different files. Moreover, I would have liked to have a 
    # more general implementation, so that I could have calculated the different values for 
    # any Jacobian. But, the time limitation and other tasks did not give me the time to fully
    # polish my work.
    
    # should have used more simplifies!
    
    print(tau_2)
    tau_2 = tau_2.subs(((t1, 0), (t3, 0), (t3dot, 0), (t1dot, 0), (diff_t1, 0), (t1_hat, 0), (diff_t3_hat, 0), (t3_hat, 0), (time_diff_t3, 0), (time_diff_t1, 0), (diff_t3_hat, 0), (diff_t2_hat, 0), (c1_n, 0), (c3_n, 0)))
    print(tau_2)
  

def pprint(args):
    sp.pretty_print(args)
    
if __name__ == '__main__':
    main()
