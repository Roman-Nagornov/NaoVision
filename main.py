"""
    The module that controls all aspects of drawing experiments:
    - preparing for drawing
    - drawing
    - end of experiment
"""
sys.path.append('./naoqi/')
from naoqi import ALProxy

if __name__ == "__main__":
    PORT = 9559
    IP = "192.168.1.33"
    motion = ALProxy("ALMotion", IP, PORT)
    posture = ALProxy("ALRobotPosture", IP, PORT)

    # Please, place nao before the paper sheet.
    posture.goToPosture("StandZero", 0.5)


    posture.goToPosture("StandInit", 0.5)
    pass