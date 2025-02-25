from xmlrpc.client import Boolean
import constantes as cst
import roboticstoolbox as rtb
from Motor import *
import warnings
import numpy as np
from MyFifo import *

class Yokobo():
    def __init__(self, unit=cst.RADIAN, nbrMotors=cst.NUMBER_OF_MOTOR, mass=cst.MASS):
        self.robot = cst.ROBOT
        self.unit = unit
        self.mass = mass

        self.timer = 0
        self.duration = -1
        self.preTimer = 0
        
        self.motors = []
        for _ in range(nbrMotors):
            self.motors.append(Motor(unit=unit))
        self.eeVelocity = []
        self.eeAcceleration = []
        self.eeJerk = []
        self.eeEnergy = []

        self.firstCallPad = True
        
        self.OFF_LIGHT = list(cst.PALETTE)[0]
         
        self.color = [self.OFF_LIGHT]
        self.colorFifo = MyFifo(cst.LIGHT_SAMPLING_SIZE)
        self.colorFifo.add(self.OFF_LIGHT)
        self.luminosity = [128]
        self.luminosityFifo = MyFifo(cst.LIGHT_SAMPLING_SIZE)
        self.luminosityFifo.add(128)    

# -- OPERATORS
    def __str__(self) -> str:
        text = ""
        for mot in self.motors:
            text += 'M' + str(mot.id) + ' : ' + str(mot.position()) + '    '

        text += "LIGHT(" + self.color[-1] + ", " + str(self.luminosity[-1]) + ")"
        return text

    def __len__(self):
        return len(self.motors)

# -- GETTER SETTER
    def position(self): # angular position of each motor
        pos = []
        for mot in self.motors:
            pos.append(mot.position())
        return pos

    def velocity(self): # angular velocity of each motor
        pos = []
        for mot in self.motors:
            pos.append(mot.velocity())
        return pos

    def trajectory(self, motorId=0):
        if motorId > len(self.motors):
            raise ValueError("ID out of range (max: " + str(len(self.motors)) + ")")

        if motorId != 0:
            return self.motors[motorId-1].trajectory()

        traj = []
        lengthTraj = -1
        tempL = -1
        for mot in self.motors:
            tempL = len(mot.trajectory())
            if tempL > lengthTraj:
                lengthTraj = tempL

        for mot in self.motors:
            temp = cst.fitList(mot.trajectory(), lengthTraj)            
            traj.append(temp)            

        return traj, lengthTraj
    
    def units(self, motorId=0, sep=" "):
        if motorId > len(self.motors) or motorId < 0:
            raise ValueError("ID out of range (max: " + str(len(self.motors)) + ")")

        if motorId != 0:
            return self.motors[motorId-1].unit
        else:
            units = ""
            for mot in self.motors:
                units += sep + mot.unit
            return units[1:]

    def averageEndEffectorVelocity(self, samplingSize=cst.AVERAGE_SIZE):
        '''
            Calculate the average end effector velocity over the last *samplingSize* data in the end effector frame, return -1 if there a not enough data
            Output form:
            [
                v_x: velocity over x axis in m/s,
                v_y: velocity over y axis in m/s,
                v_z: velocity over z axis in m/s
            ]
        '''
        return self._averageEndEffector(data=self.eeVelocity, samplingSize=samplingSize)

    def averageEndEffectorAcceleration(self, samplingSize=cst.AVERAGE_SIZE):
        '''
            Calculate the average end effector acceleration over the last *samplingSize* data in the end effector frame, return -1 if there a not enough data
            Output form:
            [
                a_x: acceleration over x axis in m/s²,
                a_y: acceleration over y axis in m/s²,
                a_z: acceleration over z axis in m/s²
            ]
        '''        
        return self._averageEndEffector(data=self.eeAcceleration, samplingSize=samplingSize)

    def averageEndEffectorJerk(self, samplingSize=cst.AVERAGE_SIZE):
        '''
            Calculate the average end effector jerk over the last *samplingSize* data in the end effector frame, return -1 if there a not enough data
            Output form:
            [
                j_x: jerk over x axis in m/s^3,
                j_y: jerk over y axis in m/s^3,
                j_z: jerk over z axis in m/s^3
            ]
        '''
        return self._averageEndEffector(data=self.eeJerk, samplingSize=samplingSize)

    def averageEndEffectorEnergy(self, samplingSize=cst.AVERAGE_SIZE):
        '''
            Calculate the average end effector kinematic energy over the last *samplingSize* data in the end effector frame, return -1 if there a not enough data
            Output form:
            [
                Ek_x: kinematic energy over x axis in J,
                Ek_y: kinematic energy over y axis in J,
                Ek_z: kinematic energy over z axis in J
            ]
        '''
        return self._averageEndEffector(data=self.eeEnergy, samplingSize=samplingSize)


    def magnitudeEndEffectorVelocity(self, average=False):
        if not isinstance(average, Boolean):
            data = self.averageEndEffectorVelocity(average)
        else:
            if not self.eeVelocity:
                data = [0,0,0]
            else:
                data = self.eeVelocity[-1]
        return self._magnitudeEndEffector(data)

    def magnitudeEndEffectorAcceleration(self, average=False):
        if average:
            data = self.averageEndEffectorAcceleration()
        else:
            if not self.eeAcceleration:
                data = [0,0,0]
            else:
                data = self.eeAcceleration[-1]
        return self._magnitudeEndEffector(data)

    def magnitudeEndEffectorJerk(self, average=False):
        if average:
            data = self.averageEndEffectorJerk()
        else:
            if not self.eeJerk:
                data = [0,0,0]
            else:
                data = self.eeJerk[-1]

        return self._magnitudeEndEffector(data)

    def magnitudeEndEffectorEnergy(self, average=False):
        if average:
            data = self.averageEndEffectorEnergy()
        else:
            if not self.eeEnergy:
                data = [0,0,0]
            else:
                data = self.eeEnergy[-1]
        return self._magnitudeEndEffector(data)

# -- PRIVATE FUNCTIONS  
    def _endEffectorVelocity(self): # (v_x, v_y, v_z)
        J = self.robot.jacob0(self.robot.q)
        #print(J * np.atleast_2d(self.velocity()).T)
        try:
            self.eeVelocity.append(np.dot(J, np.atleast_2d(self.velocity()).T)[0:3])
        except Exception as e:
            print(e)
        
        self._endEffectorEnergy()
        self._endEffectorAcceleration()

    def _endEffectorAcceleration(self): # (a_x, a_y, a_z)
        if len(self.eeVelocity)<2:
            return
        else:
            self.eeAcceleration.append((self.eeVelocity[-1]-self.eeVelocity[-2])/self.duration)
        self._endEffectorJerk()

    def _endEffectorJerk(self): # (j_x, j_y, j_z)
        if len(self.eeAcceleration)<2:
            return
        else:
            self.eeJerk.append((self.eeAcceleration[-1]-self.eeAcceleration[-2])/self.duration)

    def _endEffectorEnergy(self): # (E_x, E_y, E_z)
        self.eeEnergy.append((self.mass/2) * np.power(self.eeVelocity[-1], 2))

    def _averageEndEffector(self, data, samplingSize=cst.AVERAGE_SIZE):
        if len(data) < samplingSize:
            return False
        else:
            return sum(data[-samplingSize:])/samplingSize

    def _magnitudeEndEffector(self, data):
        if isinstance(data, Boolean) and data == False:
            return False
        else:
            return float(np.sqrt(np.power(data[0],2)+np.power(data[1],2)+np.power(data[2],2)))


# -- PUBLIC FUNCTIONS
    def reset(self):
        for mot in self.motors:
            mot.reset()

        self.eeVelocity = []
        self.eeAcceleration = []
        self.eeJerk = []
        self.eeEnergy = []

        self.timer = 0
        self.duration = -1
        self.preTimer = 0

        self.luminosity = [128]
        self.color = [self.OFF_LIGHT]
        self.colorFifo.reset()
        self.colorFifo.add(self.OFF_LIGHT)
        self.luminosityFifo.reset()
        self.luminosityFifo.add(128)

        self.firstCallPad = True
        

    def move(self, positions):
        if len(positions) != len(self.motors):
            raise ValueError("Number of position does not match the number of motors.")

        for i in range(len(self.motors)):
            try:
                firstMotorOriginFlag = True
                if i==1:
                    if self.motors[0].positionsList[-1]>=(cst.MOTOR_ORIGIN[self.motors[0].unit][0] + cst.EPSILON_MOTOR_1[self.motors[0].unit]) or\
                        self.motors[0].positionsList[-1]<=(cst.MOTOR_ORIGIN[self.motors[0].unit][0] - cst.EPSILON_MOTOR_1[self.motors[0].unit]):
                        
                        firstMotorOriginFlag = True

                self.motors[i].move(positions[i], firstMotorOriginFlag) 
            except ValueError as e: # out of range of the motor
                raise ValueError(e)

        self.timer = time.perf_counter()
        self.duration = self.timer - self.preTimer
        self.robot.q = self.position()
        self._endEffectorVelocity()
        self.preTimer = self.timer
       
    def pad(self):
        pleasure, arousal, dominance = 0, 0, 0

        jerk = self.magnitudeEndEffectorJerk(True)
        energy = self.magnitudeEndEffectorEnergy(True)
        m2 = self.position()[1]

        PAD_MIN = min(cst.PAD)
        PAD_MAX = max(cst.PAD)

        #print("acce: " + str(self.magnitudeEndEffectorAcceleration(True)))
        if jerk != False:
            if jerk > cst.JERK_MAX and not self.firstCallPad: 
                # Since we cannot compute the maximum jerk, and avoid the first point that has high jerk
                cst.JERK_MAX = jerk     
            pleasure = -1 * (((PAD_MAX-PAD_MIN)*jerk/cst.JERK_MAX) + PAD_MIN)            

        if energy != False:
            arousal = ((PAD_MAX-PAD_MIN)*energy/cst.ENERGY_MAX) + PAD_MIN

        M2_MIN = cst.MOTOR_MIN[self.unit][1]
        M2_MAX = cst.MOTOR_MAX[self.unit][1]
        dominance = ((PAD_MAX-PAD_MIN)/(M2_MAX - M2_MIN)) * m2 + ((PAD_MIN * M2_MAX - PAD_MAX * M2_MIN)/(M2_MAX - M2_MIN))   
        
        # SECURITY CHECKS
        if pleasure     > PAD_MAX:   pleasure = PAD_MAX
        if pleasure     < PAD_MIN:   pleasure = PAD_MIN
        if arousal      > PAD_MAX:   arousal = PAD_MAX
        if arousal      < PAD_MIN:   arousal = PAD_MIN
        if dominance    > PAD_MAX:   dominance = PAD_MAX
        if dominance    < PAD_MIN:   dominance = PAD_MIN

        self.firstCallPad = False
        return [pleasure, arousal, dominance]

    def light(self, color, luminosity):
        colorChange = False
        closeColor = False
        palToList = list(cst.PALETTE)
        newColor = palToList[color]
        if self.color[-1] != newColor and self.color[-1] != self.OFF_LIGHT and newColor != self.OFF_LIGHT: # to avoid brutal change of colour, beside switch on/off
            colorChange = True
        previousColorId = palToList.index(self.color[-1])
        if color == (previousColorId-1) or color == (previousColorId+1):
            closeColor = True

        self.color.append(newColor)
        self.colorFifo.append(newColor)

        outOfRange = False
        self.luminosity.append(self.luminosity[-1] + luminosity)
        if self.luminosity[-1] < 0:
            self.luminosity[-1] = 0
            outOfRange = True
        elif self.luminosity[-1] > 255:
            self.luminosity[-1] = 255
            outOfRange = True

        self.luminosityFifo.append(self.luminosity[-1])
        return colorChange, outOfRange, closeColor
    