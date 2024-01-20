from stepperDriver import Stepper
from servo import Servo

stepperH = Stepper(0,1,2,3,-1)
servoV = Servo(28)

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
        steps = calculateStepperDiff(float(msg))
        print("Moving horizontally by ", steps)
        stepperH.run(steps)
    elif topic == b'pico/mot/vert':
        pos = float(msg)
        print("Moving vertically to ", pos)
        servoV.move(pos)
