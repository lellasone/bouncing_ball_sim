from system import System
from tkinter import *
import sympy as sym
import numpy as np
import threading
import time
root = Tk()

class gui():

    def __init__(self):
        self.VERBOSITY = 3 
        self.width = 410 # width of canvas. 
        self.height = 410 # height of canvas. 
        #self.setup_simulation()
        #self.run_simulation() 
        self.sys = System()
        self.state = [0,200,np.pi/4,0]
        self.dt = 0.01 # how long to jump each simulation time step.

        self.g_cw = self.sys.build_G(self.sys.build_R(-np.pi),[self.width/2,self.height,0])
        root = self.create_gui()
        draw = threading.Thread(target = self.animate_canvas)
        simulate = threading.Thread(target = self.simulate_system)
        
        draw.start()
        simulate.start()
        root.mainloop()

    def simulate_system(self):
        """ 
        #TODO: add docstring. 
        This function will throttle if the simulation time begins to run faster
        then the real-life time.
        """
        #TODO: Document.
        t = 0
        traj = [self.state]
        state_old = self.state # Used in computing the impact equation
        while(True):
            # Used for throttling 
            end = time.time() + self.dt
            
            t += self.dt 
            col = self.sys.check_collisions()
            if col > 0 :
                s = self.sys.collision_update(state_old, col)
            else:
                s = self.sys.integrate(self.state,self.dt)

            if(True):
                state_old = self.state
                self.state = s
                traj.append(s)
                log(s, 3)
            
            # If we have time left in the loop, sleep till next draw needed.
            if end - time.time() -.0001 > 0:
                time.sleep(end - time.time())

 
    def animate_canvas(self):
        """! Thread for drawing the canvas based on the system state. 
        This function is meant to be called in a seperate thread after the 
        system and gui have both been set up, but before the tkinter mainloop
        has been called.

        It will re-draw the canvas to reflect the system state vector at 60hz
        or as close to that as the computer's speed allows. This is the thread
        that is responsible for actually showing the system's motion on screen.
        """
        delay = 1.0/60
        while(True):
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

        subs = {self.sys.q[0]:s[0],
                self.sys.q[1]:s[1],
                self.sys.q[2]:s[2],
                self.sys.q[3]:s[3]}
         
        vec_z = sym.Matrix([0,0,0,1])
                                         
        # draw the ball.
        b0 = (self.g_cw*self.sys.g_wb*self.sys.g_bb0*vec_z).subs(subs)
        b1 = (self.g_cw*self.sys.g_wb*self.sys.g_bb1*vec_z).subs(subs)
        b2 = (self.g_cw*self.sys.g_wb*self.sys.g_bb2*vec_z).subs(subs)
        b3 = (self.g_cw*self.sys.g_wb*self.sys.g_bb3*vec_z).subs(subs)
        ball_new = self.canvas.create_polygon(b0[0], b0[1],
                                   b1[0], b1[1],
                                   b2[0], b2[1],
                                   b3[0], b3[1])
        self.log("Ball Drawn", 4)

         # draw the plate.
        p0 = (self.g_cw*self.sys.g_wp*self.sys.g_pp0*vec_z).subs(subs)
        p1 = (self.g_cw*self.sys.g_wp*self.sys.g_pp1*vec_z).subs(subs)
        p2 = (self.g_cw*self.sys.g_wp*self.sys.g_pp2*vec_z).subs(subs)
        p3 = (self.g_cw*self.sys.g_wp*self.sys.g_pp3*vec_z).subs(subs)
        plate_new = self.canvas.create_polygon(p0[0], p0[1],
                                   p1[0], p1[1],
                                   p2[0], p2[1],
                                   p3[0], p3[1])
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

        subs = {self.sys.q[0]:s[0],
                self.sys.q[1]:s[1],
                self.sys.q[2]:s[2],
                self.sys.q[3]:s[3]}
         
        vec_z = sym.Matrix([0,0,0,1])
         
        # draw the boarder. 
        e0 = self.g_cw*self.sys.g_we*self.sys.g_ee0*vec_z.subs(subs)
        e1 = self.g_cw*self.sys.g_we*self.sys.g_ee1*vec_z.subs(subs)
        e2 = self.g_cw*self.sys.g_we*self.sys.g_ee2*vec_z.subs(subs)
        e3 = self.g_cw*self.sys.g_we*self.sys.g_ee3*vec_z.subs(subs)
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
        start = Button(buttons,text="start")
        stop = Button(buttons, text="pause")
        reset = Button(buttons,text="Reset")
        start.grid(row = 0,column = 0,padx = 5)
        stop.grid(row = 0, column = 1, padx = 5)
        reset.grid(row = 0, column = 2, padx = 5)


        state = Frame(interface)
        label0 = Label(state, text = "Ball X")
        label1 = Label(state, text = "Ball Y")
        label2 = Label(state, text = "Ball Angle")
        label3 = Label(state, text = "Plate Angle")
        s0 = Entry(state, width = 10)
        s1 = Entry(state, width = 10)
        s2 = Entry(state, width = 10)
        s3 = Entry(state, width = 10)
        label0.grid(column = 0, row = 0)
        label1.grid(column = 1, row = 0)
        label2.grid(column = 2, row = 0)
        label3.grid(column = 3, row = 0)
        s0.grid(column = 0, row = 1)
        s1.grid(column = 1, row = 1)
        s2.grid(column = 2, row = 1)
        s3.grid(column = 3, row = 1)

        state.pack()
        buttons.pack()

        self.canvas = Canvas(display, height = self.height, width = self.width)
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
