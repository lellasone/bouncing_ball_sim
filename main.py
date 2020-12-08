from system import System
from tkinter import *
import sympy as sym
import numpy as np
import threading
import time
from simple_pid import PID
root = Tk()

class gui():

    def __init__(self):
        
        self.sys = System()
        self.VERBOSITY = 2

        # Set the visual paramiters.  
        self.scale = 500 # scaling factor between simulated system and pixels
        boarder = 0.05 # Width of the canvas edge border as a portion of the 
                       # canvas dimensions
        self.width = self.sys.el*(1 + boarder) # width of canvas. 
        self.height = self.sys.el*(1 + boarder) # height of canvas. 
        self.initial_state =   [-0.1,self.height/2,np.pi/5,np.pi/20, .25,0,-1,-1]
        self.state = self.initial_state
        self.dt = 0.005 # how long to jump each simulation time step.

        self.g_cw = self.sys.build_G(self.sys.build_R(-np.pi),
                                    [self.width/2,
                                    self.height - self.height*boarder*0.5,
                                    0])


        # Vars relating to the closed loop control
        self.forces = [0,0]
        self.control_enable = IntVar() 
        self.inner = PID(0, 0, 0, 0) # These are not safe defaults, change
        self.outer = PID(0, 0, 0, 0) # during system bringup.  
        self.outer.output_limits = (-np.pi/8, np.pi/8)
        self.wind_scale = -0.5
        self.plate_goal = 0 


        root = self.create_gui()
        stop = threading.Event()
        draw = threading.Thread(target = self.animate_canvas, args = (stop,))
        simulate = threading.Thread(target = self.simulate_system, args = (stop,))
        self.run = False # Should the simulation continue

        
        try:
            draw.start()
            simulate.start()
            root.mainloop()
        finally:
            stop.set()  

    def compute_control_forces(self, running):
        
        if running:  
            # Handle outer loop
            self.auto_mode = True
            try:
                self.outer.Kp = self.Kp_o.get() 
                self.outer.Ki = self.Ki_o.get()
            except:
                self.log("Invalid Input",2)
            self.sample_time = self.dt
            self.plate_goal = self.outer(self.state[0])
          

              # Handle inner loop 
            self.auto_mode = True
            self.inner.setpoint = self.plate_goal
            try:
                self.inner.Kp = self.Kp_i.get() 
                self.inner.Kd = self.Kd_i.get()
            except:
                self.log("Invalid Input",2)           
            self.sample_time = self.dt
            self.forces[1] = self.inner(self.state[3])

        else:
            self.auto_mode = False  

    def simulate_system(self, kill_thread):
        """ 
        #TODO: add docstring. 
        This function will throttle if the simulation time begins to run faster
        then the real-life time.
        """
        #TODO: Document.
        t = self.dt
        traj = [self.state]
        state_old = self.state # Used in computing the impact equation
        while(not kill_thread.is_set()):
            # Used for throttling 
            end = time.time() + self.dt
            
           
            if(self.run):
                self.compute_control_forces(True) 
                # Check Collisions or update sim
                col = self.sys.check_collisions(self.state, self.dt)
                if col > 0 :
                    self.log("COLLISION #{}".format(col), 2)
                    s = self.sys.collision_update(state_old, col)
                else:
                    s = self.sys.integrate(self.state,self.dt, self.forces)
                self.log(type(s))
                
                #Record new states     
                t += self.dt 
                state_old = self.state
                self.state = s
                traj.append(s)
                self.set_state_gui(s)
                self.log(s, 3)
            else:
                self.state[0:4] = self.get_state_gui()
                self.compute_control_forces(False) 
            # If we have time left in the loop, sleep till next draw needed.
            if end - time.time() -.0001 > 0:
                time.sleep(end - time.time())   


    def get_state_gui(self):
        s = [np.float64(self.s0.get()),
             np.float64(self.s1.get()), 
             np.float64(self.s2.get()), 
             np.float64(self.s3.get())]
        return(s)

    def set_state_gui(self, s):
        
        self.s0.set(s[0])
        self.s1.set(s[1])
        self.s2.set(s[2])
        self.s3.set(s[3])
        
    def pause(self):
            self.run = False

    def start(self):
            self.run = True
    
    def reset(self):
            self.state = self.initial_state
            self.set_state_gui(self.state)
            self.run = False
 
    def animate_canvas(self, kill_thread):
        """! Thread for drawing the canvas based on the system state. 
        This function is meant to be called in a seperate thread after the 
        system and gui have both been set up, but before the tkinter mainloop
        has been called.

        It will re-draw the canvas to reflect the system state vector at 60hz
        or as close to that as the computer's speed allows. This is the thread
        that is responsible for actually showing the system's motion on screen.
        """
        delay = 1.0/60
        while(not kill_thread.is_set()):
            end = time.time() + delay
            ball_new, plate_new =self.update_canvas(self.state)
            self.canvas.delete(self.ball)
            self.canvas.delete(self.plate)
            self.ball = ball_new
            self.plate = plate_new
            # If we have time left in the loop, sleep till next draw needed.
            if end - time.time() -.0001> 0:
                time.sleep(end - time.time())
 
    def update_canvas(self,s):
        """! ns A tuple containing the ball and plate objects.
        """
        scale = self.scale # Honestly this is just so the lines fit

        subs = {self.sys.q[0]:s[0],
                self.sys.q[1]:s[1],
                self.sys.q[2]:s[2],
                self.sys.q[3]:s[3]}
         
        vec_z = sym.Matrix([0,0,0,1])
                                         
        # draw the ball.
        b0 = (self.g_cw*self.sys.g_wb*self.sys.g_bb0*vec_z).subs(subs) * scale
        b1 = (self.g_cw*self.sys.g_wb*self.sys.g_bb1*vec_z).subs(subs) * scale
        b2 = (self.g_cw*self.sys.g_wb*self.sys.g_bb2*vec_z).subs(subs) * scale
        b3 = (self.g_cw*self.sys.g_wb*self.sys.g_bb3*vec_z).subs(subs) * scale
        ball_new = self.canvas.create_polygon(b0[0], b0[1],
                                   b1[0], b1[1],
                                   b2[0], b2[1],
                                   b3[0], b3[1])
        self.log("Ball Drawn", 4)

         # draw the plate.
        p0 = (self.g_cw*self.sys.g_wp*self.sys.g_pp0*vec_z).subs(subs) * scale
        p1 = (self.g_cw*self.sys.g_wp*self.sys.g_pp1*vec_z).subs(subs) * scale
        p2 = (self.g_cw*self.sys.g_wp*self.sys.g_pp2*vec_z).subs(subs) * scale
        p3 = (self.g_cw*self.sys.g_wp*self.sys.g_pp3*vec_z).subs(subs) * scale
        plate_new = self.canvas.create_polygon(p0[0], p0[1],
                                   p1[0], p1[1],
                                   p2[0], p2[1],
                                   p3[0], p3[1])

        self.forces[0] = np.float64(self.wind.get())*self.wind_scale
        return(ball_new, plate_new) 
 
    def draw_canvas(self,s):
        """! Draw a given state vector on the canvas. 
        Pre-condition: Tkinter interface must be set up. 
                       self.sys must be defined. 

        @param s initial system state vector to display.
        @returns A tuple containing the ball and plate objects.
        """
        # Remove old canvas.
        self.canvas.delete("all") 

        scale = self.scale
        subs = {self.sys.q[0]:s[0],
                self.sys.q[1]:s[1],
                self.sys.q[2]:s[2],
                self.sys.q[3]:s[3]}
         
        vec_z = sym.Matrix([0,0,0,1])
         
        # draw the boarder. 
        e0 = self.g_cw*self.sys.g_we*self.sys.g_ee0*vec_z.subs(subs) * scale
        e1 = self.g_cw*self.sys.g_we*self.sys.g_ee1*vec_z.subs(subs) * scale
        e2 = self.g_cw*self.sys.g_we*self.sys.g_ee2*vec_z.subs(subs) * scale
        e3 = self.g_cw*self.sys.g_we*self.sys.g_ee3*vec_z.subs(subs) * scale
        self.canvas.create_line(e0[0],e0[1],e1[0],e1[1],
                                e2[0],e2[1],
                                e3[0],e3[1],
                                e0[0],e0[0],
                                width = 5)
        self.log("Boarder Drawn", 4)
        
        ball, plate = self.update_canvas(s)  

        return(ball, plate) 
    
    def create_gui(self):
        interface = Frame(root)
        display = Frame(root)


        # Build Interface
        buttons = Frame(interface)
        start = Button(buttons,text="start",command = self.start)
        stop = Button(buttons, text="pause", command = self.pause)
        reset = Button(buttons,text="Reset", command = self.reset)
        self.wind = Scale(buttons, from_=-100, to=100, orient = HORIZONTAL)
        start.grid(row = 0,column = 0,padx = 5)
        stop.grid(row = 0, column = 1, padx = 5)
        reset.grid(row = 0, column = 2, padx = 5)
        self.wind.grid(row = 0, column = 3, padx = 5)


        state = Frame(interface)
        
        self.s0 = StringVar(state)
        self.s1 = StringVar(state)
        self.s2 = StringVar(state)
        self.s3 = StringVar(state)
        label0 = Label(state, text = "Ball X")
        label1 = Label(state, text = "Ball Y")
        label2 = Label(state, text = "Ball Angle")
        label3 = Label(state, text = "Plate Angle")
        s0 = Entry(state, width = 10, textvariable = self.s0)
        s1 = Entry(state, width = 10, textvariable = self.s1)
        s2 = Entry(state, width = 10, textvariable = self.s2)
        s3 = Entry(state, width = 10, textvariable = self.s3)
        label0.grid(column = 0, row = 0)
        label1.grid(column = 1, row = 0)
        label2.grid(column = 2, row = 0)
        label3.grid(column = 3, row = 0)
        s0.grid(column = 0, row = 1)
        s1.grid(column = 1, row = 1)
        s2.grid(column = 2, row = 1)
        s3.grid(column = 3, row = 1)
        self.set_state_gui(self.state)

        # Build Controller Interface
        control = Frame(interface)
        self.Kp_i = DoubleVar(control)
        self.Kd_i = DoubleVar(control)
        self.Kp_o = DoubleVar(control)
        self.Ki_o = DoubleVar(control)
        label_kpi = Label(control, text = "KP - Inner Loop")
        label_kdi = Label(control, text = "KD - Inner Loop")
        label_kpo = Label(control, text = "KP - Outer Loop")
        label_kio = Label(control, text = "Ki - Outer Loop")
        enable = Checkbutton(control, 
                             text = "Enable", 
                             variable = self.control_enable) 
        kpi = Entry(control, width = 15, textvariable = self.Kp_i)
        kdi = Entry(control, width = 15, textvariable = self.Kd_i)
        kpo = Entry(control, width = 15, textvariable = self.Kp_o)
        kio = Entry(control, width = 15, textvariable = self.Ki_o)
        label_kpi.grid(column = 1, row = 0) 
        label_kdi.grid(column = 2, row = 0) 
        label_kpo.grid(column = 3, row = 0) 
        label_kio.grid(column = 4, row = 0) 
        enable.grid(column = 0, row = 1, rowspan = 2)
        kpi.grid(column = 1, row = 1) 
        kdi.grid(column = 2, row = 1) 
        kpo.grid(column = 3, row = 1) 
        kio.grid(column = 4, row = 1) 
        
        state.pack()
        control.pack() 
        buttons.pack()
        

        self.canvas = Canvas(display, 
                             height = self.height * self.scale, 
                             width = self.width * self.scale)
        self.canvas.pack()


        interface.grid(column = 1, row = 1)
        display.grid(column = 1, row = 0)
        self.ball, self.plate = self.draw_canvas(self.state)


        return(root)

    def log(self, msg, priority = 3):
        """! Verbosity aware way to print messages to the terminal. 
        @param msg, the string to be printed. 
        @param priority, how important the message is. Recommended scale is from 0 to 3.
        """
        if(self.VERBOSITY >= priority):
            print(msg)
        
        


if __name__ =="__main__":
    main = gui()
