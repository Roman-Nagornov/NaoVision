import cv2
import ShowProjection as sp
import numpy as np
import NAO_cs_converter as cs_conv

import pylab as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Must be > 0
DISTORT_TRESHOLD = 1


def get_paper_sheet_POSE(img_client):
    # Initialize video client
    cv2.namedWindow("find_paper_sheet")
    # Try to get the first frame
    if img_client.isOpened():
        rval, tmp_frame = img_client.getPic()
    else:
        rval = False

    # If we've got a new frame
    if rval:
        frame_num = 0
        while frame_num < DISTORT_TRESHOLD:
            # Undistort frame
            frame = cv2.undistort(tmp_frame, sp.cam_intr_mtx, sp.cam_dist, None, sp.newcameramtx)
            is_detected, paper_vertexes, cmr_rvec, cmr_tvec = sp.detect_paper_sheet(frame)
            rval, tmp_frame = img_client.getPic()

            frame_num += 1
            # Exit on 'q'

            cv2.putText(frame,
                        '#frame:%.f' % (frame_num),
                        (0, 20),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (0, 0, 0))
            cv2.imshow("find_paper_sheet", frame)
            print frame_num
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyWindow("find_paper_sheet")
    else:
        is_detected = False
        paper_vertexes = None
        cmr_rvec = None
        cmr_tvec = None
    return is_detected, paper_vertexes, cmr_rvec, cmr_tvec


if __name__ == "__main__":
    #vc = cv2.VideoCapture(0)
    PORT = 9559
    IP = "192.168.1.35"
    import retrieve_bottom_cmr_frame as ret_img
    img_client = ret_img.NAOcam(IP, PORT)
    is_detected, paper_vertexes, cmr_rvec, cmr_tvec = get_paper_sheet_POSE(img_client)

    if not is_detected:
        print "Object is not detected"
        exit(1)

    cmr_rmtx, jacobian = cv2.Rodrigues(cmr_rvec)

    # Convert paper sheet coordinates from cmr coordinate system to robot coordinate system
    NAO_CS = cs_conv.bottom_cmr_converter(IP, PORT)
    world_frame2body_rmtx, world_frame2body_tvec = NAO_CS.get_world_frame2img_cs_transform(cmr_rmtx.T, cmr_tvec)
    #####################################################
    # Visualisation
    #######################################################

    coub_axis = [[], [], []]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', aspect='equal')
    #print vertexes_cmr_cs

    import test_visualisation as tv
    # ax.plot(vertexes_cmr_cs[0], vertexes_cmr_cs[1], vertexes_cmr_cs[2], c='0.5', marker='.', alpha=0.5)
    # ax.scatter([0], [0I, [0], c='r', marker='.', s=10, alpha=0.5)

    minmax_list = tv.show_cs(np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), np.array([0., 0., 0.]), ax)
    for ind in xrange(3):
        coub_axis[ind] += [minmax_list[0][ind], minmax_list[1][ind]]


    """
    minmax_list = tv.show_cs(world_frame2body_rmtx, np.array(world_frame2body_tvec).T[0], ax)
    for ind in xrange(3):
        coub_axis[ind] += [minmax_list[0][ind], minmax_list[1][ind]]
    """
    nao_mtxs_list = NAO_CS.getMtxDict()
    tmp_transform = nao_mtxs_list['world_frame2head_transform']
    tmp_transform[:3, :3] = tmp_transform[:3, :3].T
    mtxs_list_world_frame = [tmp_transform]
    """,


                             nao_mtxs_list['world_frame2head_transform'] * nao_mtxs_list['head2bottom_cmr_transform'],
                             nao_mtxs_list['world_frame2head_transform'] * nao_mtxs_list['head2bottom_cmr_transform'] * nao_mtxs_list['bottom_cmr2img_cs_transform']]
                            """
    for transform in mtxs_list_world_frame:
        rmtx_tmp, tvec_tmp = NAO_CS.transform2rmtx_tvec(transform)
        minmax_list = tv.show_cs(rmtx_tmp, tvec_tmp, ax)
        for ind in xrange(3):
            coub_axis[ind] += [minmax_list[0][ind], minmax_list[1][ind]]
    import test_contours as tc
    #contours_list_3d_model_cs = list()
    # for contour in tc.test_contour:
    #     contour_3d_model_cs = pr2t3.convert_drawing_pnts_2d_3d(np.array(contour, dtype='float64'), map(float, tc.shape), (sp.paper_sheet_vertexes[1, 0], sp.paper_sheet_vertexes[2, 1]))
    #     #print contour_3d_model_cs
    #     contour_3d_cmr_cs, minmax_list = pr2t3.sheet_crds_to_cmr_crds(cmr_rvec, cmr_tvec, contour_3d_model_cs)
    #     #print contour_3d_cmr_cs
    #     contour_3d_cmr_cs = contour_3d_cmr_cs.T
    #     ax.plot(contour_3d_cmr_cs[0], contour_3d_cmr_cs[1], contour_3d_cmr_cs[2], c='0.5', marker='.', alpha=0.5)

    # Magic workaround for matplotlib aspect ratio bug.
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
    #######################################################
    pass
