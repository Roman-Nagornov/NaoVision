import cv2
import ShowProjection as sp
import project_2d_to_3d as pr2t3
import pylab as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def show_cs(rmtx, tvec, ax):
    scale = 0.1
    colors = ['r', 'g', 'b']
    #rmtx, jacobian = cv2.Rodrigues(rvec)
    rmtx = np.array(rmtx)
    for ind in xrange(3):
        ax.plot([tvec[0], tvec[0] + scale * rmtx[ind, 0]],
                [tvec[1], tvec[1] + scale * rmtx[ind, 1]],
                [tvec[2], tvec[2] + scale * rmtx[ind, 2]], c=colors[ind])

    pass

if __name__ == "__main__":
    # Initialize video client
    cv2.namedWindow("find_paper_sheet")
    vc = cv2.VideoCapture(0)

    # Try to get the first frame
    if vc.isOpened():
        rval, tmp_frame = vc.read()
        # Undistort frame
    else:
        rval = False

    if rval:

        is_detected = False

        frame_num = 0
        while frame_num < 20:
            frame = cv2.undistort(tmp_frame, sp.cam_intr_mtx, sp.cam_dist, None, sp.newcameramtx)
            is_detected, paper_vertexes, cmr_rvec, cmr_tvec = sp.detect_paper_sheet(frame)
            rval, tmp_frame = vc.read()
            # Undistort frame

            if is_detected:
                frame_num += 1
            # Exit on 'q'
            cv2.imshow("find_paper_sheet", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyWindow("find_paper_sheet")

        vertexes_cmr_cs = pr2t3.sheet_crds_to_cmr_crds(cmr_rvec, cmr_tvec, sp.paper_sheet_vertexes)
        vertexes_cmr_cs = vertexes_cmr_cs.T

        cmr_rmtx, jacobian = cv2.Rodrigues(cmr_rvec)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', aspect='equal')

        ax.plot(vertexes_cmr_cs[0], vertexes_cmr_cs[1], vertexes_cmr_cs[2], c='0.5', marker='.', alpha=0.5)
        ax.scatter([0], [0], [0], c='r', marker='.', s=10, alpha=0.5)
        show_cs(np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), np.array([0., 0., 0.]), ax)
        show_cs(cmr_rmtx.T, cmr_tvec.T[0], ax)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    pass