# -*- encoding: UTF-8 -*-
# Get an image from NAO. Display it and save it using PIL.
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
# Python Image Library
import Image
sys.path.append('./naoqi/')
from naoqi import ALProxy
import vision_definitions

class NAOcam:
    """
    Take 640x480 image from NAO
    """
    videoClient = None
    camProxy = None

    def __init__(self, IP, PORT):
        """
        initialize videoClient and camProxy
        """
        self.camProxy = ALProxy("ALVideoDevice", IP, PORT)
        resolution = 2        # VGA
        colorSpace = 11     # RGB
        cameraID = 1        # Lower camera
        self.videoClient = self.camProxy.subscribe("python_client", resolution, colorSpace, 5)
        # Select camera.
        self.camProxy.setParam(vision_definitions.kCameraSelectID, cameraID)
        pass

    def isOpened(self):
        # Make checking is image retrieved
        return True

    def getPic(self):
        """
        Returns pic as numpy RGB array
        """
        naoImage = self.camProxy.getImageRemote(self.videoClient)
        imageWidth = naoImage[0]
        imageHeight = naoImage[1]
        array = naoImage[6]
        img = np.array(Image.fromstring("RGB", (imageWidth, imageHeight), array))
        # Make checking is image retrieved
        return True, img

    def showImg(self, img=None):
        """
        Show img made by object of this class. If (img == None) then take pic and show it
        """
        if (img == None):
            img = self.getPic()
        plt.imshow(img)
        plt.show()
        pass

    def saveImg(self, fnameJPG, img=None):
        """
        Save img made by object of this class. If (img == None) then take pic and save it
        fnameJPG - name of image is string '*.jpg'
        """
        if (img == None):
            img = self.getPic()
        plt.imshow(img)
        plt.imsave(fnameJPG, img)
        pass

    def __del__(self):
        self.camProxy.unsubscribe(self.videoClient)
        pass


if __name__ == '__main__':
    IP = "192.168.1.43"    # Replace here with your NaoQi's IP address.
    PORT = 9559

    naoCam = NAOcam(IP, PORT)
    for ind in xrange(14):
        time.sleep(1)
        naoCam.saveImg("chessboard" + str(ind) + '.jpg')
