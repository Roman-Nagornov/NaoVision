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
    minmax_list = [[tvec[0], tvec[1], tvec[2]], [tvec[0] + scale * rmtx[ind, 0],
                  tvec[1] + scale * rmtx[ind, 1],
                  tvec[2] + scale * rmtx[ind, 2]]]
    return minmax_list

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

        # minmax_x = list()
        # minmax_y = list()
        # minmax_z = list()
        coub_axis = [[], [], []]
        vertexes_cmr_cs, minmax_list = pr2t3.sheet_crds_to_cmr_crds(cmr_rvec, cmr_tvec, sp.paper_sheet_vertexes)
        for ind in xrange(3):
            coub_axis[ind] += [minmax_list[0][ind], minmax_list[1][ind]]
        print coub_axis

        cmr_rmtx, jacobian = cv2.Rodrigues(cmr_rvec)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', aspect='equal')

        vertexes_cmr_cs = vertexes_cmr_cs.T
        ax.plot(vertexes_cmr_cs[0], vertexes_cmr_cs[1], vertexes_cmr_cs[2], c='0.5', marker='.', alpha=0.5)
        ax.scatter([0], [0], [0], c='r', marker='.', s=10, alpha=0.5)
        minmax_list = show_cs(np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), np.array([0., 0., 0.]), ax)
        for ind in xrange(3):
            coub_axis[ind] += [minmax_list[0][ind], minmax_list[1][ind]]
        minmax_list = show_cs(cmr_rmtx.T, cmr_tvec.T[0], ax)
        for ind in xrange(3):
            coub_axis[ind] += [minmax_list[0][ind], minmax_list[1][ind]]

        # Magic workaround for the Matplolib bug.
        # coub_axis = list()
        #for ind in xrange(3):
            # coub_axis.append(np.array(map(lambda x: x[:, ind].max(), chain_position_list)
            #     + map(lambda x: x[:, ind].min(), chain_position_list)
            #     + map(lambda x: x[ind].max(), target_end_effector_pos)
            #     + map(lambda x: x[ind].max(), target_end_effector_pos)))
        # + [target_end_effector_pos[ind]])
        # Create cubic bounding box to simulate equal aspect ratio
        # max_range = np.array([coub_axis[0].max() - coub_axis[0].min(), coub_axis[1].max() - coub_axis[1].min(), coub_axis[2].max() - coub_axis[2].min()]).max()
        coub_axis = np.array(coub_axis)
        max_range = np.array(map(lambda axis: axis.max() - axis.min(), coub_axis)).max()
        grid_list = list()
        for ind in xrange(3):
            curr_axis_grid = 0.5 * max_range * np.mgrid[-1: 2: 2, -1: 2: 2, -1: 2: 2][ind].flatten() + 0.5 * (coub_axis[ind].max() + coub_axis[ind].min())
            grid_list.append(curr_axis_grid)
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(grid_list[0], grid_list[1], grid_list[2]):
            ax.plot([xb], [yb], [zb], 'w')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    pass