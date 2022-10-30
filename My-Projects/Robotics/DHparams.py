import numpy as np
from sympy import symbols, Matrix, pi, sin, cos, eye, pretty_print
from functools import partial


def main():

    # number of joints
    joints = 3
    variable_names = ['theta', 'd', 'a', 'alpha']

    # theta(1)a(1)d(1)alpha(1), ..., theta(n)a(n)d(n)alpha(n)
    # placed in array respectively
    DH_variables = [f'{i}'.join(variable_names)
                    + str(i) for i in range(1, joints + 1)]
    # print(DH_variables)

    # all variables are saved as symbols (symbolic)
    symbolic_DH = [symbols(DH) for DH in DH_variables]
    # print(symbolic_DH)

    # theta, d, a, alpha
    # specified DH-table from mandatory assignment 1

    DH_table = [[symbols('theta1'), symbols('L1'), 0, pi/2],
                [symbols('theta2'), 0, symbols('L2'), 0],
                [symbols('theta3'), 0, symbols('L3'), 0]]

    FK = forward_kinematics(DH_table)
    t_1 = -90
    t_2 = 30
    t_3 = -45
    org_angles = [t_1, t_2, t_3]

    FK_numpy = forward_kinematics_numpy([t_1, t_2, t_3], FK)

    # --------------------------------------------------------------------
    # Print-statements for matrix, coordinate and DH-parameters
    # --------------------------------------------------------------------
    print("\n** The generalized DH-table for the CrustCrawler robot **")
    t, d, a, alpha = symbols('theta d a alpha')
    DH_copy = Matrix(DH_table.copy())
    pretty_print([t, d, a, alpha])
    pretty_print(DH_copy)

    print("\n** The complete homogenous transformation matrix **")
    pretty_print(FK)

    print("\nTask 1A) -----------------------------------------------------------------------")
    print("\n** The homogenous transformation matrix for\n  ", "theta_1 = ",
          t_1, " theta_2 = ", t_2, " and theta_3 = ", t_3, " **")
    round4 = partial(round, ndigits=4)
    pretty_print(FK_numpy.applyfunc(round4))
    x, y, z = FK_numpy.applyfunc(round4)[0:3, 3]
    print(f'\n** Coordinates from forward kinematics **\n'
          f'\nwith theta_1 = {t_1} degrees, theta_2 = {t_2} degrees\n'
          f' and theta_3 = {t_3} degrees'
          f' yield:\n \n{x = }\n{y = }\n{z = }')
    print("\n--------------------------------------------------------------------------------")
    # --------------------------------------------------------------------
    x, y, z = [np.float64(float) for float in FK_numpy[0:3, 3]]
    inv_angles = find_inverse([x, y, z])
    inv_t1, inv_t2, inv_t3 = [np.rad2deg(angle) for angle in inv_angles]


    print("\nTask 1B) -----------------------------------------------------------------------")
    print("\n** The inverse kinematics function **")
    print("\nThe cartesian coordinates of the tip used in the inverse kinematics problem are\n"
          "\n\tx =", x, "\n\ty = ", y, "\n\tz = ", z )
    print("\n and the inverse kinematics function yields the degrees of the angles")
    print("\n\tTheta_1 = ", round(inv_t1, 4), "\n\tTheta_2 = ", -round(inv_t2, 4),
          "\n\tTheta_3 = ", round(inv_t3, 4))
    print("\n--------------------------------------------------------------------------------")

    check_correctness([inv_t1, -inv_t2, inv_t3], [x, y, z], FK, org_angles)

    solution_sets = four_soultions(org_angles, [inv_t1, inv_t2, inv_t3], FK, [0, -323.9033, 176.6988])

def four_soultions(angles, inv_angles, FK, cart_cord):

    L_1 = 100.9
    L_2 = 222.1
    L_3 = 136.2
    x, y, z = cart_cord

    # theta_1:
    t_1 = np.arctan2(y, x)
    t_1_r = np.arctan2(y, x) + np.pi

    # theta_3:
    r_1 = np.sqrt(x ** 2 + y ** 2)
    r_2 = z - L_1
    cos_t3 = ((r_1**2 + r_2**2 - L_2**2 - L_3**2) / (2 * L_2 * L_3))
    D = cos_t3
    t_3_ED = np.arctan2(D, np.sqrt(1 - D**2))
    t_3_EU = np.arctan2(D, -np.sqrt(1 - D**2))

    t2 = np.arctan2(np.sqrt(r_1**2), r_2) - np.arctan2(L_2 + L_3 * np.cos(t_3_EU), L_3 * np.sin(t_3_EU))
    t_2_ED = np.arctan2(r_1, r_2) - np.arctan2(L_2 + L_3 * np.cos(t_3_ED), L_3 * np.sin(t_3_ED))
    t_2_EU = np.arctan2(r_1, r_2) - np.arctan2(L_2 + L_3 * np.cos(t_3_EU), L_3 * np.sin(t_3_EU))


    t_1, t_1_r = np.rad2deg(t_1), np.rad2deg(t_1_r)
    t_2_ED, t_2_EU = -np.rad2deg(t_2_ED), np.rad2deg(t_2_EU)
    t_3_ED, t_3_EU = np.rad2deg(t_3_ED), np.rad2deg(t_3_EU)

    print("\nTask 1D) -----------------------------------------------------------------------")
    print("\t** The four solutions to the inverse kinematics problem **")

    round4 = partial(round, ndigits=4)
    print("\n\t-- Elbow UP configuration --")
    FK_numpy0 = forward_kinematics_numpy([-90, 30, -45], FK)
    x_0, y_0, z_0 = FK_numpy0.applyfunc(round4)[0:3, 3]
    print(f'\n\tThe angles theta_1 = -90, theta_2 = 30 and theta_3 = -45 degrees\n\t'
          f'yield the coordinates {x_0 = }, {y_0 = } and {z_0 = }.')

    FK_numpy3 = forward_kinematics_numpy([90, 90 + 60, 45], FK)
    x_3, y_3, z_3 = FK_numpy3.applyfunc(round4)[0:3, 3]
    print(f'\n\tThe angles theta_1 = 90, theta_2 = 150 and theta_3 = 45 degrees\n\t'
          f'yield the coordinates {x_3 = }, {y_3 = } and {z_3 = }.')

    print("\n\t-- ELBOW UP CONFIGURATION --")
    FK_numpy1 = forward_kinematics_numpy([t_1, t_2_ED, t_3_ED], FK)
    x_1, y_1, z_1 = FK_numpy1.applyfunc(round4)[0:3, 3]
    print(f'\n\tThe angles theta_1 = ', round(t_1,4),' theta_2 = ', round(t_2_ED,4), ' and theta_3 = ', round(t_3_ED,4)
          ,' degrees\n\t'
          f'yield the coordinates {x_1 = }, {y_1 = } and {z_1 = }.')

    FK_numpy2 = forward_kinematics_numpy([t_1_r, (-t_2_ED + 180), -t_3_ED], FK)
    x_2, y_2, z_2 = FK_numpy2.applyfunc(round4)[0:3, 3]
    print(f'\n\tThe angles theta_1 = ', round(t_1_r,4), ' theta_2 = ', round(-t_2_ED+180,4), ' and theta_3 = '
          , round(-t_3_ED,4), ' degrees\n\t'
          f'yield the coordinates {x_2 = }, {y_2 = } and {z_2 = }.')

    print("\n--------------------------------------------------------------------------------")

def check_correctness(inverse_angles, forward_coordinates, FK, org_angles):
    inv_t1, inv_t2, inv_t3 = inverse_angles
    f_x, f_y, f_z = [round(np.float64(coordinate),4) for coordinate in forward_coordinates]
    inv_FK_numpy = forward_kinematics_numpy(inverse_angles, FK)
    i_x, i_y, i_z = [(round(np.float64(float), 4)) for float in inv_FK_numpy[0:3, 3]]

    if (f_x == i_x) and (f_y == i_y) and (f_z == i_z) :
        print("\nTask 1C) -----------------------------------------------------------------------")
        print("\n** The inverse kinematics gave the same results as for that with the forward\n"
              "kinematics case **")
        print("\n\tForward kinematics with degrees theta_1 = ", org_angles[0], ", theta_2 = ", org_angles[1],
              "\n\tand theta_3 = ", org_angles[2], " yield the results: ")
        print(f'\n\t{f_x = }\n\t{f_y = }\n\t{f_z = }')
        print("\n\tInverse kinematics with degrees theta_1 = ", round(inverse_angles[0],4), ", theta_2 = ",
              round(inverse_angles[1],4),
              "\n\tand theta_3 = ", round(inverse_angles[2],4))
        print(f'\n\t{i_x = }\n\t{i_y = }\n\t{i_z = }')
    else:
        print("** Woops! An error occured, and the inverse and the forward kinematics did not yield the"
              "same coordinate results.")
    print("\n--------------------------------------------------------------------------------")

def forward_kinematics(DH_table):
    # unpack values and
    A_matrices = [A_matrix(*DH) for DH in DH_table]

    FK = eye(4)
    for A in A_matrices: FK @= A
    return FK

def A_matrix(theta_i, d_i, a_i, alpha_i):
    theta, d, a, alpha = symbols('theta d a alpha')

    rot_z = Matrix([[cos(theta), -sin(theta), 0, 0],
                    [sin(theta), cos(theta), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    trans_z = Matrix([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, d],
                      [0, 0, 0, 1]])

    trans_x = Matrix([[1, 0, 0, a],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

    rot_x = Matrix([[1, 0, 0, 0],
                    [0, cos(alpha), -sin(alpha), 0],
                    [0, sin(alpha), cos(alpha), 0],
                    [0, 0, 0, 1]])

    A = rot_z @ trans_z @ trans_x @ rot_x

    # substitute values
    A = A.subs([(theta, theta_i), (d, d_i), (a, a_i), (alpha, alpha_i)])

    return A

def forward_kinematics_numpy(joint_variables, forwardK):
    # joint_variables = [theta_1, theta_2, theta_3]
    d1 = 100.9
    d2 = 222.1
    d3 = 136.2

    t1, t2, t3, l1, l2, l3 = symbols('theta1 theta2 theta3 L_1 L_2 L_3')

    a1, a2, a3 = [np.deg2rad(degrees) for degrees in joint_variables]
    DH_table = Matrix([[t1, l1, 0, pi / 2],
                [t2, 0, l2, 0],
                [t3, 0, l3, 0]])
    DH_table = DH_table.subs([(t1, a1), (t2, a2), (t3, a3), (l1, d1), (l2, d2), (l3, d3)])
    DH_table = DH_table.tolist()
    A_matrices = [A_matrix(*DH) for DH in DH_table]
    FK = eye(4)

    for A in A_matrices: FK @= A

    return FK

def find_inverse(cart_cord):
    L_1 = 100.9
    L_2 = 222.1
    L_3 = 136.2
    x, y, z = cart_cord

    # theta_1:
    t1 = np.arctan2(y, x)

    # theta_3:
    r_power_2 = x**2 + y**2
    s = z - L_1
    a_2 = L_2
    a_3 = L_3
    cos_t3 = ( (r_power_2 + s**2 - a_2**2 - a_3**2) / (2*a_3*a_2))
    t3 = np.arctan2(cos_t3, np.sqrt(1 - cos_t3**2))

    # theta_2:
    t2 = np.arctan2(np.sqrt(r_power_2), s) - np.arctan2(a_2 + a_3*np.cos(t3), a_3*np.sin(t3))

    return [t1, t2, t3]

if __name__ == '__main__':
    main()