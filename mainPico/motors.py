from stepperDriver import Stepper
from servo import Servo

stepperH = Stepper(0,1,2,3,-1)
servoV = Servo(28)
servoV.move(90)

stepsPerDegree=64/5.625
oldPos=0

def degreeToSteps(degrees):
    return degrees * stepsPerDegree

def calculateStepperDiff(newPos):
    global oldPos
    posDifferenceInDegrees = newPos-oldPos
    oldPos = newPos
    return degreeToSteps(posDifferenceInDegrees)

def callback(topic, msg):
    print("In motor callback, topic: ", topic)
    if topic == b'pico/mot/hor':
        requestedPosition = float(msg)
        if requestedPosition > 180 or requestedPosition < -180:
            print("Requested V position out of range, ignoring:", requestedPosition)
            return
        steps = calculateStepperDiff(requestedPosition)
        print("Moving horizontally by ", steps)
        stepperH.run(steps)
        
    elif topic == b'pico/mot/vert':
        pos = 90-float(msg)
        if pos > 45 or pos < -45:
            print('Requested H position out of range, ignoring: ', pos)
            return
        print("Moving vertically to ", pos)
        servoV.move(pos)
