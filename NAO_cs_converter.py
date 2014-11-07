import math
import numpy as np
from naoqi import ALProxy
from showCoordSys import showCS

class CSconverter:

    def invTransfMtx(self, TransMtx):
        """ type(TransMtx) == numpy.matrix """
        R = TransMtx[:3, :3]
        tr = TransMtx[:3, 3]
        res1 = np.bmat([R.T, -R.T * tr])
        res2 = np.bmat([np.zeros(3), [1]])
        resMtx = np.bmat([[res1], [res2]])
        return resMtx

    def rotX(self, phi):
        result = np.mat(
            [[1., 0., 0., 0.],
            [0., math.cos(phi), -math.sin(phi), 0.],
            [0., math.sin(phi), math.cos(phi), 0.],
            [0., 0., 0., 1.]]
        )
        return result

    def rotY(self, phi):
        result = np.mat(
            [[math.cos(phi), 0., math.sin(phi), 0.],
            [0., 1., 0., 0.],
            [-math.sin(phi), 0., math.cos(phi), 0.],
            [0., 0., 0., 1.]]
        )
        return result

    def rotZ(self, phi):
        result = np.mat(
            [[math.cos(phi), -math.sin(phi), 0., 0.],
            [math.sin(phi), math.cos(phi), 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]]
        )
        return result

    pass

class NAOvisCS(CSconverter):
    motionProxy = None
    camMtxRelHeadRMtx = None
    imgCSrelCamMtx = None
    CSpace = None

    #HeadRMtx = None
    HeadRMtxRelWMtx = None


    def __init__(self, IP, PORT, _CSpace=1):
        """

        """
        self.CSpace = _CSpace
        camYaw = 1.2 * math.pi / 180.
        XcamTrans, YcamTrans, ZcamTrans = 0.05871, 0., 0.06364    # in meters
        self.camMtxRelHeadRMtx = np.mat([[math.cos(camYaw), 0., math.sin(camYaw), XcamTrans],
                          [0., 1., 0., YcamTrans],
                          [-math.sin(camYaw), 0., math.cos(camYaw), ZcamTrans],
                          [0., 0., 0., 1.]])
        tauZcam = -math.pi / 2.
        tauXcam = -math.pi / 2.
        self.imgCSrelCamMtx = self.rotZ(tauZcam) * self.rotX(tauXcam)
        try:
            self.motionProxy = ALProxy("ALMotion", IP, PORT)
        except Exception, e:
            print "Error when creating ALMotion proxy:"
            print str(e)
            exit(1)
        pass

    def getNAOvisCSdata(self):
        #HeadYaw = self.motionProxy.getAngles('HeadYaw', True)[0]
        #HeadPitch = self.motionProxy.getAngles('HeadPitch', True)[0]
        #self.HeadRMtx = self.rotZ(HeadYaw) * self.rotY(HeadPitch)
        self.HeadRMtxRelWMtx = np.mat(self.motionProxy.getTransform('Head', self.CSpace, True)).reshape((4, 4))
        pass

    def update(self):
        self.getNAOvisCSdata()
        pass

    def cam2CSpace(self, vect):
        """
        vect.shape = (3, 1)
        """
        if self.HeadRMtxRelWMtx == None:
            self.getNAOvisCSdata()
        hvect = np.append(vect, [[1.]], axis=0)
        return self.HeadRMtxRelWMtx * self.camMtxRelHeadRMtx * self.imgCSrelCamMtx * hvect #* self.HeadRMtx

    def getMtxDict(self):
        if self.HeadRMtxRelWMtx == None:
            self.getNAOvisCSdata()
        result = {'HeadRMtxRelWMtx': self.HeadRMtxRelWMtx, #'HeadRMtx': self.HeadRMtx,
                'camMtxRelHeadRMtx': self.camMtxRelHeadRMtx, 'imgCSrelCamMtx': self.imgCSrelCamMtx}
        return result

    def saveNAO_VisCoordData(self):
        pass

    def loadNAO_VisCoordData(self):
        pass

    pass


if __name__ == "__main__":
    IP = "nao.local"    # Replace here with your NaoQi's IP address.
    PORT = 9559
    NAO_CS = NAOvisCS(IP, PORT)
    print NAO_CS.cam2CSpace(np.mat([[1.], [2.], [3.]]))
