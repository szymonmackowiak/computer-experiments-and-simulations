from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def module(vec):
    module = (vec.x ** 2 + vec.y ** 2 + vec.z ** 2) ** 0.5
    return module

def newMatrix2d(a, i, j):
    return np.array([[a for k in range(i)] for u in range(j)])

def distanceVectorMatrix(PartSys):
    dist_vec_mat = newMatrix2d(PVector3(0, 0, 0), PartSys.n, PartSys.n)
    for i in range(PartSys.n):
        for j in range(PartSys.n):
            if j == i:
                dist_vec_mat[i][j] = PVector3(0, 0, 0)
            elif j > i:
                dist_vec_mat[i][j] = PartSys.partList[i].pos.sub(PartSys.partList[j].pos)
            #else:
                #dist_vec_mat[j][i].x = (-1)*dist_vec_mat[j][i].x
                #dist_vec_mat[j][i].y = (-1)*dist_vec_mat[j][i].y
                #dist_vec_mat[j][i].z = (-1)*dist_vec_mat[j][i].z
                #dist_vec_mat[j][i].xyzToCoord()
    return dist_vec_mat

class PVector3(object):

    def __init__(self, x, y, z):
        self.coord = np.array([x, y, z])
        self.x = self.coord[0]
        self.y = self.coord[1]
        self.z = self.coord[2]
        
    def add(self, vec):
        self.coord = np.add(self.coord, vec.coord)
        self.x = self.coord[0]
        self.y = self.coord[1]
        self.z = self.coord[2]
        
    def mult(self, scalar):
        self.coord = self.coord*scalar
        self.x = self.coord[0]
        self.y = self.coord[1]
        self.z = self.coord[2]

    def sub(self, vec):
        self.coord = np.subtract(self.coord, vec.coord)
        self.x = self.coord[0]
        self.y = self.coord[1]
        self.z = self.coord[2]

    def xyzToCoord(self):
        self.coord = np.array([self.x, self.y, self.z])

    def norm(self):
        m = module(self)
        self.x = self.coord[0]/m
        self.y = self.coord[1]/m
        self.z = self.coord[2]/m
        self.coord = np.array([self.x, self.y, self.z])        

    def distanceMatrix(pos0, pos1, pos2):
        temp_list = [pos0, pos1, pos2]
        dist_mat = np.array([[0, 0, 0], \
                             [0, 0, 0], \
                             [0, 0, 0]])
        mat = distanceVectorMatrix(pos0, pos1, pos2)
                            
        for i in range(3):
            for j in range(3):
                if j == i:
                    dist_mat[i][j] = 0
                elif j > i:
                    dist_mat[i][j] = length.distanceVectorMatrix(pos0, pos1, pos2)[i][j]
                else:
                    dist_mat[i][j] = (-1)*dist_mat[j][i]
        return dist_mat

class Particle(object):

    def __init__(self, position, velocity):
        self.pos = PVector3(position.x, position.y, position.z)
        self.vel = PVector3(velocity.x, velocity.y, velocity.z)
        self.acc = PVector3(0, 0, 0)

        self.posOld = self.pos
        self.velOld = self.vel
        self.accOld = self.acc

    def simpleStep(self, dt):
        self.velOld.x = self.vel.x
        self.velOld.y = self.vel.y
        self.velOld.z = self.vel.z
        self.velOld.xyzToCoord()

        self.posOld.x = self.pos.x
        self.posOld.y = self.pos.y
        self.posOld.z = self.pos.z
        self.posOld.xyzToCoord()

        self.vel.x = self.vel.x + self.acc.x * dt
        self.vel.y = self.vel.y + self.acc.y * dt
        self.vel.z = self.vel.z + self.acc.z * dt
        self.vel.xyzToCoord()

        self.pos.x = self.pos.x + self.vel.x * dt + 0.5 * self.acc.x * dt * dt
        self.pos.y = self.pos.y + self.vel.y * dt + 0.5 * self.acc.y * dt * dt
        self.pos.z = self.pos.z + self.vel.z * dt + 0.5 * self.acc.z * dt * dt
        self.pos.xyzToCoord()
    
    def updateVerlet(self, dt):
        
        self.acc.add(mult(forceVector(self, another), (1/m)))
        
        self.pos.x = 2 * self.pos.x - self.posOld.x + self.acc.x * dt * dt 
        self.pos.y = 2 * self.pos.y - self.posOld.y + self.acc.y * dt * dt
        self.pos.z = 2 * self.pos.z - self.posOld.z + self.acc.z * dt * dt
        self.pos.xyzToCoord()

        self.vel.x = (self.pos.x - self.posOld.x) / (2 * dt)
        self.vel.y = (self.pos.y - self.posOld.y) / (2 * dt)
        self.vel.z = (self.pos.z - self.posOld.z) / (2 * dt)
        self.vel.xyzToCoord()

        self.posOld.x = self.pos.x
        self.posOld.y = self.pos.y
        self.posOld.z = self.pos.z
        self.posOld.xyzToCoord()
              
    def display(self):
        # Make data
        u = np.linspace(0, 2 * np.pi, 12)
        v = np.linspace(0, np.pi, 12)
        diameter = 1
        xs = diameter * np.outer(np.cos(u), np.sin(v))
        ys = diameter * np.outer(np.sin(u), np.sin(v))
        zs = diameter * np.outer(np.ones(np.size(u)), np.cos(v))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        ax.plot_surface(xs+self.pos.x, ys+self.pos.y, zs+self.pos.z, color='b')
        ax.auto_scale_xyz([-diameter, diameter], [-diameter, diameter], [-diameter, diameter])
        plt.show()
        

class ParticleSystem(object):
    
    def __init__(self, org, n):
        self.n = n
        self.origin = PVector3(org.x, org.y, org.z)
        self.partList = np.array([])

    def addParticle(self, part):
        self.partList = np.append(self.partList, part)

    def cubArrange(self, dist, length):
        self.dist = dist
        self.length = length
        
        for i in range(0, self.n):
            x = dist*np.mod(i, length)
            y = dist*np.mod(np.floor((i)*(length**(-1))),length);
            z = dist*np.mod(np.floor((i)*(length**(-2))),length);
            
            position = PVector3(x, y, z)
            velocity = PVector3(0, 0, 0)

            atom = Particle(position, velocity)

            self.addParticle(atom)

    def update(self, dt):
        for i in range(0, self.n):
            self.partList[i].updateVerlet(dt)

    def display(self):
        u = np.linspace(0, 2 * np.pi, 12)
        v = np.linspace(0, np.pi, 12)
        diameter = 1
        xs = diameter * np.outer(np.cos(u), np.sin(v))
        ys = diameter * np.outer(np.sin(u), np.sin(v))
        zs = diameter * np.outer(np.ones(np.size(u)), np.cos(v))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        for i in range(0, self.n):
            x = self.partList[i].pos.x
            y = self.partList[i].pos.y
            z = self.partList[i].pos.z
            ax.plot_surface(xs+x, ys+y, zs+z, color='b')

        ax.auto_scale_xyz([0, 1.5*self.length], [0, 1.5*self.length], [0, 1.5*self.length])
        plt.show()

    
#################################################
#           THE SIMULATION STARTS HERE          #
#################################################

origin = PVector3(0, 0, 0)
uklad = ParticleSystem(origin, 64)
uklad.cubArrange(2, 4)
uklad.display()
uklad.update(0.001)
uklad.display()
uklad.update(0.001)
uklad.display()
uklad.update(0.001)
