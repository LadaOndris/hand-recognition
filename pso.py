
"""
PSO

"""

import numpy as np


class Particle:
    
    def __init__(self, posRange, velRange):
        
        self.posRange = np.array(posRange)
        self.velRange = np.array(velRange)
        self.position = np.array(np.random.randint(posRange[0], posRange[1], size=(2,)), dtype=np.float32)
        self.velocity = np.array(np.random.randint(velRange[0], velRange[1], size=(2,)), dtype=np.float32)
        self.bestPosition = self.position

    def update(self, swarmBestPosition, target,
               inertia = 0.4,
               cp = 2.05,
               cg = 2.05):
        oldPosition = self.position
        self.position = np.add(self.position, self.velocity)
        self.velocity = inertia * self.velocity + \
            cp * np.random.random() * (self.bestPosition - oldPosition) + \
            cg * np.random.random() * (swarmBestPosition - oldPosition)
        
        self.setToMinMax(self.position, self.posRange)
        self.setToMinMax(self.velocity, self.velRange)
        
        if self.comparePositions(self.position, self.bestPosition, target) < 1:
            self.bestPosition = self.position
    
    
    def comparePositions(self, pos1, pos2, target):
        dist1 = np.linalg.norm(pos1-target)
        dist2 = np.linalg.norm(pos2-target)
        if dist1 < dist2:
            return -1
        elif dist1 > dist2:
            return 1
        else:
            return 0
    
    def setToMinMax(self, array, interval):
        for i in range(len(array)):
            if array[i] < interval[0]:
                array[i] = interval[0]
            elif array[i] > interval[1]:
                array[i] = interval[1]
        
class PSO:
    
    def __init__(self, 
                 particles = 500,
                 posRange = (-10000, 10000),
                 velRange = (-10,10)):
        self.bestPosition = None
        self.particlesCount = particles
        self.posRange = posRange
        self.velRange = velRange
        self.createParticles()
        
    def createParticles(self):
        self.particles = []
        for i in range(self.particlesCount):
            self.particles.append(Particle(self.posRange, self.velRange))
    
    def updateBestPosition(self, target):
        if self.bestPosition is None:
            self.bestPosition = self.particles[0].bestPosition
            
        for particle in self.particles:
            if particle.comparePositions(particle.bestPosition, self.bestPosition, target) < 0:
                self.bestPosition = particle.bestPosition
        
        
    def optimize(self, target, iterations=10, callback=None):
        self.updateBestPosition(target)
        for i in range(iterations):
            for particle in self.particles:
                particle.update(self.bestPosition, target)
            self.updateBestPosition(target)
            callback({'best': self.bestPosition})
        
    
def step(params):
    for key, value in params.items():
        print(key, value)
    
pso = PSO()
pso.optimize(target=(900, -900), iterations=100, callback = step)