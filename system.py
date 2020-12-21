"""! This file contains most of the math used for the simulation.

The System class contains all of the symbolic math required to define the 
system and compute state derivatives for a particular state. However, it does
not have any provisions for integration/simulation and will thus need to be 
paired with external code for simulating and visualizing the system.

No state information is stored in the System class so it can be re-used for new
simulations without issue.  
"""
import sympy as sym
import numpy as np
from sympy import cos, sin, tan, S

class System():
    """Simulates the simplified case of two rigid legs 'walking' in responce to externally applied torques.
    
    TODO: Add how-to-use. 
    """
    def __init__(self, s = [0,0,0,0,0,0,0,0]):
        """! Sets various constants, defines variables and defines the relationships between frames in the system. 
        """
        self.VERBOSITY = 3 # Lower value to see less readouts. 
        ### SET UP VARIABLES ###
        
        t, s = sym.symbols('t, s')
        self.t = t
        
        # Lets define the system's known constants. 
        self.el = 1 # Length of each side of the operating frame. 
        self.ph = self.el * 0.2 # Height of plate center/pivot. 
        self.pw = self.el * 0.1 # Width of the plate. 
        self.pl = self.el * 0.7 # Length of the plate. 
        self.bl = self.el/10 # side length of square ball. 
        
        self.G = 9.8 # Force of gravity
        self.Mb = 1
        self.Mp = 10
       
        #TODO: Collision distance tolerance for absolute collisions
        self.col_scale = 4
        
        # Lets set up our state vector.
        x = sym.Function('x')(self.t)
        y = sym.Function('y')(self.t)
        theta = sym.Function('theta')(self.t)
        phi = sym.Function('phi')(self.t)
        
        self.q = sym.Matrix([x, y, theta, phi])
        self.qd = self.q.diff(self.t)
        self.qdd = self.qd.diff(self.t)
        
        # Lets define some relationships between frames. 
        self.define_transforms()
        
        
        ### SET UP THE SYSTEM OF EQUATIONS ###
        # Build our legrangian
        self.LG = self.define_LG()       
        
        # Lets define the collision equations. 
        self.define_collisions()
        self.define_collision_updates()
        
        # Lets define some external forces. 
        Fx, Fphi = sym.symbols('F_x, \tou')
        self.F = sym.Matrix([Fx, 0, 0, Fphi])
        

                
        ### SOLVE THE SYSTEM OF EQUATIONS ###
        # Constrained eueler legrange equation.
        EL = sym.Eq(self.compute_EL(self.LG).T, self.F) 
        
        self.log("Computing Solutions", 0)
        self.sols = self.solve_EL(EL)
        self.log("Accelerations")
        self.log(self.sols[0][self.qdd[0]])
        self.log(self.sols[0][self.qdd[1]])
        self.log(self.sols[0][self.qdd[2]])
        self.log(self.sols[0][self.qdd[3]])
        self.qddl = self.lambdify_EL(self.sols[0])
        self.log("Solutions Computed",0)


    def define_collision_updates(self):
        """! 
        """
        self.log("Defining Collision Updates", 2)
        q = self.q
        qd  = self.qd
        self.q_dum = [sym.symbols('x'),
                     sym.symbols('y'),
                     sym.symbols('theta'),
                     sym.symbols('phi')]
        self.qd_dum = [sym.symbols('v_x'),
                      sym.symbols('v_y'),
                      sym.symbols('omega_b'),
                      sym.symbols('omega_p')]
        
        self.qd_dum_p = [sym.symbols('v_x^+'),
                        sym.symbols('v_y^+'),
                        sym.symbols('omega_b^+'),
                        sym.symbols('omega_p^+')]
             
        self.log(self.q_dum, 3)
        sym_subs = {q[0]:self.q_dum[0],
                    q[1]:self.q_dum[1], 
                    q[2]:self.q_dum[2], 
                    q[3]:self.q_dum[3], 
                    qd[0]:self.qd_dum[0], 
                    qd[1]:self.qd_dum[1], 
                    qd[2]:self.qd_dum[2], 
                    qd[3]:self.qd_dum[3]}
    
        sym_subs_p = {self.qd_dum[0]:self.qd_dum_p[0],
                      self.qd_dum[1]:self.qd_dum_p[1], 
                      self.qd_dum[2]:self.qd_dum_p[2], 
                      self.qd_dum[3]:self.qd_dum_p[3]}
        lamb = sym.symbols('lambda')
        # Lets create some quantities we use each time without change. 
        dldqd = sym.Matrix([self.LG]).jacobian(self.qd).subs(sym_subs)
        H = self.compute_H(self.LG, self.q).subs(sym_subs)        
        
        for i, e in enumerate(self.col_cons):
             
            c = sym.Matrix([e.equation])
            self.log(c)
            dphidq = c.jacobian(q).subs(sym_subs)
            
            # Lets build equation 1:
            eq1 = sym.Eq(dldqd.subs(sym_subs_p) - dldqd, lamb * dphidq)

            # Lets build equation 2:
            eq2 = sym.Eq(sym.Matrix([0]), H.subs(sym_subs_p) - H)
            e.col_update_equ = [eq1, eq2]
 
    def compute_collision(self,s,n):
        """! Compute the collision update for a particular collision.
        This function is meant to be called during the simulation process in 
        responce to a valid check_collisions result.
        
        This function typically takes about a second to execute.

        @param s, the system state vector on which to perform the calculation. 
                  This will generally be one 'step' before the collision was
                  detected. It cannot be the same state vector that triggered
                  the collision because the position variables are not altered.
        @param n, the index of the collision condition that triggered the 
                  collision.
        @returns the post-collision system state variable
        """
        lamb = sym.symbols('lambda')
        equations = self.col_cons[n].col_update_equ
        update_subs = {self.q_dum[0]:s[0],
                       self.q_dum[1]:s[1], 
                       self.q_dum[2]:s[2], 
                       self.q_dum[3]:s[3], 
                       self.qd_dum[0]:s[4], 
                       self.qd_dum[1]:s[5], 
                       self.qd_dum[2]:s[6], 
                       self.qd_dum[3]:s[7]}
        eq1 = equations[0].subs(update_subs) 
        eq2 = equations[1].subs(update_subs) 
        sols = sym.solve([eq1, eq2],self.qd_dum_p + [lamb], dict = True, quick = True)
        # Lets pick only the non-trivial solution. 
        sol = ''
        for i  in sols:
            if abs(i[lamb])> 10**-9:
                sol = i 
        self.log(sol) 
        return([np.float64(s[0]),
                np.float64(s[1]),
                np.float64(s[2]),
                np.float64(s[3]),
                np.float64(sol[self.qd_dum_p[0]]),
                np.float64(sol[self.qd_dum_p[1]]),
                np.float64(sol[self.qd_dum_p[2]]),
                np.float64(sol[self.qd_dum_p[3]])])

        
         
    def check_collisions(self,s, dt):
        """! Check which collision conditions are met for a system state.
        @param s, the current system state. 
        @param dt, the current system timestep.
        @returns -1 if no collision, otherwise index of collision 
        """
        
        for i, c in enumerate (self.col_cons):
            if(c.check(s, dt, i)):
                return(i)
        return(-1)
        
    def define_collisions(self):
        """! Define an array for all the system's collision conditions. 
        This function is responsible for building the CheckLine objects that
        actually define the collisions. This is what you want to edit to
        modify collision conditions, or to create new collision conditions.
        This function generates the self.cols_con variable.
        """
        self.log("defining collisions", 2)
        vec_z = sym.Matrix([0,0,0,1])
       
        # Lets snag some class variables for building CheckLine 
        G = self.G
        scale = self.col_scale
        pw = self.pw
        class CheckLine:
            def __init__(self, 
                         q,  
                         transform, 
                         impact_dir, 
                         extent_dir, 
                         limits,
                         approach_dir = 0,
                         directional_margin = pw*.5):
                """! Check if a point has impacted a line. 
                This class checks for impacts between a point and a line in the
                world. It is assumed that the line is straight and that the 
                transform provided is such that the point's position in some
                axis is zero when the impact takes place. Depending on the 
                approach_dir value either directional collisions (true if the 
                point is passed the collision axis) or absolute collisions 
                (true if the point is near the collision axis) can be used.

                Note: Where possible directional collisons (approach_dir != 0)
                      should be used, as these are more robust. 
    
                @param q, symbolic state vector of the system. 
                @param transform, the transform such that the two colliding 
                                  objects share a reference frame. 
                @param impact_dir, index of the position vector which is 
                                   perpendicular to the line being checked. 
                @param extent_dir, index on which to check if the impact point
                                   is within the line. (paralell to line)
                @param limits, array in the form [start of line, end of line]
                @param approach_dir, -1 if the object is allowed to be on the 
                                     negative side of the axis, 1 if allowed
                                     to be on the positive side of the axis.
                                     0 if both sides allowable. 
                @param directional_margin, How far behind a line to count 
                                           directional collisions. 
                """
                self.impact_dir = impact_dir
                self.extent_dir = extent_dir
                self.limits = limits
                self.approach_dir = approach_dir
                self.dir_margin = directional_margin
                self.col_update_equ = [] # This stores update equations
                self.equation = transform[impact_dir] # Used to compute update 
                self.get_position = sym.lambdify([q[0],q[1],q[2],q[3]],
                                                 transform,
                                                 "numpy")
                self.first = True # Absolute collisions don't work on first go.
                

            def check(self, s, dt, i):
                """ This function checks if a collision has occured. 
                Collisions can either be handled a point passing by a line, or 
                as a point passing within a certain variable margin of a line
                depending how directional_margin was set. 
            
                The former is more reliable at speed, the latter works with 
                non-square shapes.
                
                @param s, the current system state vecotor. 
                @param dt, the current timestep size. 
                @param i, the collision index (this is a way of passing in
                          logging values for debug purposes)
                @returns True, if a collision has occured. False otherwise. 
                """
                position = self.get_position(*list(s[0:4]))
                
                hit_line = False
                if self.approach_dir != 0: 
                    # Check for directional collison 
                    behind_line = position[self.impact_dir] * self.approach_dir < 0
                    # Check we are within a reasonable distance of the wall
                    near_line = abs(position[self.impact_dir]) < self.dir_margin
                    hit_line = near_line * behind_line
                else:
                    # Check for non-directional collision 
                    if self.first:
                        # Ignore first timestep
                        self.first = False
                        self.last_position = position
                        return(False)
                    # Compute velocity-based tolerance
                    vel = np.sqrt(np.dot(np.transpose(position - self.last_position),
                                 position - self.last_position))
                    vel = np.linalg.norm(position-self.last_position)
                    margin = (vel+G*dt) * scale  * dt 
                    # Determine if line hit
                    hit_line = (margin > abs(position[self.impact_dir]))


                loc = position[self.extent_dir]
                within_line = self.limits[0] <= loc and loc <= self.limits[1]
                return(hit_line and within_line)  
            
        C = []
        for g in [self.g_bb0, self.g_bb1, self.g_bb2, self.g_bb3]:
            loc = self.invert_G(self.g_we*self.g_ee0)*self.g_wb*g*vec_z
            C.append(CheckLine(self.q, loc, 1, 0, [0, self.el], 1))
            C.append(CheckLine(self.q, loc, 0, 1, [0, self.el], 1))     
            loc = self.invert_G(self.g_we*self.g_ee2)*self.g_wb*g*vec_z        
            C.append(CheckLine(self.q, loc, 1, 0, [-self.el, 0], -1))     
            C.append(CheckLine(self.q, loc, 0, 1, [-self.el, 0], -1))     
            loc = self.invert_G(self.g_wp*self.g_pp0)*self.g_wb*g*vec_z
            C.append(CheckLine(self.q, loc, 1, 0, [0, self.pl], -1))     
            C.append(CheckLine(self.q, loc, 0, 1, [0, self.pw], -1))      
            loc = self.invert_G(self.g_wp*self.g_pp2)*self.g_wb*g*vec_z
            C.append(CheckLine(self.q, loc, 1, 0, [-self.pl, 0], 1))     
            C.append(CheckLine(self.q, loc, 0, 1, [-self.pw, 0], 1))     

        self.col_cons = C
        self.log("Collisions Defined", 2)
    
    def define_transforms(self):
        """! Define the trasnformations between frames used by the system. 
        Must be called after the state vector has been defined. 
        """
        q = self.q
        # Lets define the core relationships between frames. 
        self.g_wp = self.build_G(self.build_R(q[3]),[0, self.ph, 0]) # Plate. 
        self.g_pp0 = self.build_G(self.build_R(0),[-self.pl/2, -self.pw/2, 0])
        self.g_pp1 = self.build_G(self.build_R(0),[-self.pl/2, self.pw/2, 0])
        self.g_pp2 = self.build_G(self.build_R(0),[self.pl/2, self.pw/2, 0])
        self.g_pp3 = self.build_G(self.build_R(0),[self.pl/2, -self.pw/2, 0])
        self.g_wb = self.build_G(self.build_R(q[2]),[q[0],q[1],0]) # Ball. 
        self.g_bb0 = self.build_G(self.build_R(0),[-self.bl/2, -self.bl/2, 0])
        self.g_bb1 = self.build_G(self.build_R(0),[-self.bl/2, self.bl/2, 0])
        self.g_bb2 = self.build_G(self.build_R(0),[self.bl/2, self.bl/2, 0])
        self.g_bb3 = self.build_G(self.build_R(0),[self.bl/2, -self.bl/2, 0])
        self.g_we = self.build_G(self.build_R(0), [0,self.el/2, 0]) # Edges.
        self.g_ee0 = self.build_G(self.build_R(0),[-self.el/2, -self.el/2, 0])
        self.g_ee1 = self.build_G(self.build_R(0),[self.el/2, -self.el/2, 0])
        self.g_ee2 = self.build_G(self.build_R(0),[self.el/2, self.el/2, 0])
        self.g_ee3 = self.build_G(self.build_R(0),[-self.el/2, self.el/2 ,0])
               

    def lambdify_EL(self, sols):
        """! Take solutions to the EL equations and lambdify them 
        @param sols, A dictionary relating the second derivatives of the state 
                     vector with functions of the state vector and it's derivative. 
        @returns the second derivative of the state vector as lambidified functions. 
        """
        # For brevity 
        q = self.q
        qd = self.qd
        qdd = self.qdd
        t = self.t
       
        # Define Arguments 
        F = [self.F[0], self.F[3]]
        q_e = [q[0], q[1], q[2], q[3], qd[0], qd[1], qd[2], qd[3]]
       
        # Lambdify! 
        x_dd = sym.lambdify([t] + q_e + F, sols[qdd[0]], "numpy")
        
        y_dd = sym.lambdify([t] + q_e + F, sols[qdd[1]], "numpy")
        
        theta_dd = sym.lambdify([t] + q_e + F, sols[qdd[2]], "numpy")
        
        phi_dd = sym.lambdify([t] + q_e + F, sols[qdd[3]], "numpy")
        
        return([x_dd, y_dd, theta_dd, phi_dd])

        
    def solve_EL(self,EL):
        """! Symbolically solve the Euler Legrange Equations.     
        """
        sols = sym.solve([EL],*self.qdd, dict = True)
        return (sols)
    

    def build_R(self, theta):
        """! Build a 2 dimensional rotation matrix given an angle theta. 
        """
        R = sym.Matrix([[cos(theta), -sin(theta), 0],
                        [sin(theta),cos(theta),   0],
                        [0,           0,          1]])
        return(R)
    
    def build_G(self, R, P):
        """! Build an SE(2) transformation given a rotation R and a translation P
        """
        g = sym.Matrix([[R[0,0],R[0,1], R[0,2],P[0]],
                        [R[1,0],R[1,1], R[1,2],P[1]],
                        [R[2,0], R[2,1],R[2,2],P[2]],
                        [0,        0,     0,       1]])
        return(g)
                                                  
    def hat_3(self, vec):
        """ Take a vector and perform the "hat" operation.
        This is often used for manipulating angular velocities. 
        @param vec a 1 x 3 vector to hat. 
        @param a 3x3 matrix containing the hatted vector. 
        """
        return(sym.Matrix([[0, -vec[2],vec[1]],
                           [vec[2],0, -vec[0]],
                           [-vec[1],vec[0], 0]]))
    
    def invert_G(self, g):
        """! Compute the inverse of a 3 dimensional transformation. 
        This computes the inverse of a transformation between frames. Note that 
        this is equivlient to a matrix inversion operation but perhaps a bit 
        less expensive to run. 
        @param g the transform to invert. 
        @returns a 4x4 matrix containing the inverse to g
        """
        R = g[0:3,0:3]
        Gi = (R.T).row_insert(3,sym.Matrix([0,0,0]).T)
        Gi = Gi.col_insert(3, self.pad(-(R.T)*g[0:3,3]))
        return(Gi)
    
    def unhat_3(self,V, check = True):
        """ Undo the hat operation. 
        This function performs checks that will fail for insufficiently 
        simplified inputs. For example the case of a diagonal element which is 
        equivilent to zero but not actually zero as passed in. If this becomes 
        an issue symplify the input before calling this function or dissable 
        the checks. 
        @param V a 3x3 matrix in the hatted format. 
        @param check if true assers will be called to confirm V is in hat(w) format. 
        @returns a 1x3 vector that could be used to build V through hatting.
        """
        self.log("Unhatting")
        self.log(V)
        if check:
            assert(V[0,2] == -V[2,0])
            assert(V[0,0] == V[1,1] == V[2,2] == 0)
            assert(V[0,1] == -V[1,0])
            assert(V[1,2] == -V[2,1])
        return(sym.Matrix([-V[1,2],V[0,2],V[1,0]]))
        
    def unhat_6(self,V):
        """ "unhat" a 4x4 matrix resulting from hatting a 1x6 vector. 
        This is used in manipulating the V^b term's in the kinetic energy 
        equation. 
        @param a vector in the follwing form:
                            [omega(hat) v]
                            [0          0]
        @returns a 1x6 vector of the form [v, omega] where v and omega are 1x3
        """
        v = V[0:3,3]
        w_hat = V[0:3,0:3]
        w = self.unhat_3(w_hat)
        return(sym.Matrix([v[0],v[1],v[2],w[0],w[1],w[2]]))
                                                  

    def pad(self, vec):
        """! Adds the extra 1 at the end of a 1x3 vector.
        """
        return(vec.row_insert(3, sym.Matrix([1])))
    
    def compute_EL(self, L):
        """ Computes the equler legrange equations using sympy. 
        
        NOTE: This version is appropriate for a forced system. 
        @param: L the system's legrangian. 
        @returns: the EL equation.
        """
        L_mat =sym.Matrix([L])
        dldq = L_mat.jacobian(self.q)
        dldqd = L_mat.jacobian(self.qd)
        
        EL =  dldqd.diff(self.t) - dldq
        return(EL)

    
    def dyn(self, s, F = [0,0]):
        """! Computes time derivative of the extended state vector.
        @param s the vector [states, state derivatives]
        @param F force vector for the system state [ball x force, plate toruqe]
        @returns s the vector [state derivatives, state second derivatives]
        """
        s_new = np.array([s[4],
                         s[5],
                         s[6],
                         s[7],
                         self.qddl[0](0, *s,*F),
                         self.qddl[1](0, *s,*F),
                         self.qddl[2](0, *s,*F),
                         self.qddl[3](0, *s,*F)])
        return(s_new)

    def collision_update(self, s, n):
        """! The dyn equivilent for collisions. 
        Currently this just wraps the compute_collision function. 
        @param s, the state vector to use in computing the update.
        @param n, which collision was detected (by index)
        @returns the new system state vector with updated derivatives 
        """
        return(self.compute_collision(s, n))

    def define_LG(self):
        """! Sets up the legrangian for the system.
        This should be called during the startup process. 
        
        @returns The Legrangian for the system.
        """
        self.log("defining the Legrangian")
       
        # Second moments for constant density rectangles 
        jb = self.Mb * (self.bl**2 + self.bl**2) /12
        jp = self.Mp * (self.pl**2 + self.pw**2) /12
        
        # Define the inertia tensors. 
        Ib = sym.Matrix([[self.Mb, 0, 0, 0, 0, 0],
                        [0, self.Mb, 0, 0, 0, 0],
                        [0, 0, self.Mb, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, jb]])
        
        Ip = sym.Matrix([[self.Mp, 0, 0, 0, 0, 0],
                        [0, self.Mp, 0, 0, 0, 0],
                        [0, 0, self.Mp, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, jp]])

        # Define the kinetic energies. 
        Vb_p = self.unhat_6(sym.simplify(self.g_wp.inv()*(self.g_wp.diff(self.t))))
        Vb_b = self.unhat_6(sym.simplify(self.g_wb.inv()*(self.g_wb.diff(self.t))))
        KE_p = 1/2 * Vb_p.T*Ip*Vb_p
        KE_b = 1/2 * Vb_b.T*Ib*Vb_b
        KE = KE_b + KE_p

        # Define the potential energies. 
        vec_zero = sym.Matrix([0,0,0,1])
        V = (self.g_wb*vec_zero)[1]*self.Mb*self.G
        self.log("Legrangian Defined")
        return(KE[0] - V)
    

    def compute_H(self, L, q):
        """ Computes the hamiltonian using sympy. 
        @param: L the system's legrangian. 
        @param: q the system's state variables.
        @returns: the EL equation.
        """
         
        qd = q.diff(self.t)
        
        L_mat = sym.Matrix([L])
        
        dldqd = L_mat.jacobian(qd)
        
        H = dldqd * qd - L_mat
        return(H)

    def integrate(self, s, dt , F):
        """! Integrate the state vector to get the next state vector. 
        This function impliments fourth order Runge-Kutta numerical
        integration. 
        @param s, the system state before time step.
        @param dt, how long each time step is. 
        @param F, External force vector for the system 
        @returns the system state at the end of the time step
        """
        f = self.dyn
        k1 = f(s, F) # Derivative at start of time step.
        k2 = f(s + dt*k1/2, F) # Derivative at halfway point.
        k3 = f(s + dt*k2/2, F) # Derivative at halfway point take 2.
        k4 = f(s + dt*k3, F) # Derivative at the end of the step.
        s_final = s + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        return(s_final)

    def log(self, msg, priority = 3):
        """! Verbosity aware way to print messages to the terminal. 
        Use this instead of print. Messages that are sent each timestep should
        have large priortiy values (low priority) and mesages that are set 
        infrequently or are important should have smaller priority values.

        @param msg, the string to be printed. 
        @param priority, how important the message is. 
                         Recommended scale is from 0 to 3.
        """
        if(self.VERBOSITY >= priority):
            print(msg)
    
        

