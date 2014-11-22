import cv2
import ShowProjection as sp
import project_2d_to_3d as pr2t3
import pylab as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


DISTORT_TRESHOLD = 2


def show_cs(rmtx, tvec, ax):
    """
        Visualise a coordinate system in default coordinate system.
        Input:
            rmtx - (3x3 numpy array) - rotation matrix.
            tvec - (1x3 numpy array) - translation vector.
            ax - matplotlib axis in which coordinate system must be visualised.
        Output:
            minmax_list - [[minx, miny, minz], [maxx, maxy, maxz] list] needed to workaround matplotlib bug.
    """
    # Scale factor for axis.
    scale = 0.1
    # Colors for XYZ
    colors = ['r', 'g', 'b']
    #rmtx, jacobian = cv2.Rodrigues(rvec)
    rmtx = np.array(rmtx)
    # Plot each axis.
    for ind in xrange(3):
        ax.plot([tvec[0], tvec[0] + scale * rmtx[ind, 0]],
                [tvec[1], tvec[1] + scale * rmtx[ind, 1]],
                [tvec[2], tvec[2] + scale * rmtx[ind, 2]], c=colors[ind])
    minmax_list = [[tvec[0], tvec[1], tvec[2]], [tvec[0] + scale * rmtx[ind, 0],
                  tvec[1] + scale * rmtx[ind, 1],
                  tvec[2] + scale * rmtx[ind, 2]]]
    return minmax_list


# def calculate_real_coordinates_of_paper_sheet(main_point, rotation_vector):
#     """
#     Translate paper coordinates to camera coordinate system
#     Input:
#     main_point - numpy 3D array coordinates of top left point
#     rotation_vector - rotation of the paper sheet relative to camera coordinate system
#     Output:
#     paper_sheet_real_crds - numpy 3D array of all paper coordinates
#     """
#     paper_sheet_real_crds = np.array([[main_point[0], main_point[1], main_point[2]],
#                                     [main_point[0], main_point[1], main_point[2]],
#                                     [main_point[0], main_point[1], main_point[2]],
#                                     [main_point[0], main_point[1], main_point[2]]], np.float32)
#     #Calculate rotation matrix from rotation vector
#     rotation_matrix = cv2.Rodrigues(rotation_vector)
#     #Calculate real paper sheet coordinates relative to camera
#     paper_sheet_real_crds = paper_sheet_real_crds + paper_vertexes
#     #Get the transformation of the paper coordinates
#     paper_sheet_real_crds = np.dot(paper_sheet_real_crds, rotation_matrix)
#
#     return paper_sheet_real_crds

def test_visualise(img_client):
    """
        Function for test visualisation.
    :param img_client:
    :return:
    """
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
            cv2.imshow("find_paper_sheet", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyWindow("find_paper_sheet")

        coub_axis = [[], [], []]
        vertexes_cmr_cs, minmax_list = pr2t3.sheet_crds_to_cmr_crds(cmr_rvec, cmr_tvec, sp.paper_sheet_vertexes)
        for ind in xrange(3):
            coub_axis[ind] += [minmax_list[0][ind], minmax_list[1][ind]]

        cmr_rmtx, jacobian = cv2.Rodrigues(cmr_rvec)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', aspect='equal')
        #print vertexes_cmr_cs
        vertexes_cmr_cs = vertexes_cmr_cs.T

        ax.plot(vertexes_cmr_cs[0], vertexes_cmr_cs[1], vertexes_cmr_cs[2], c='0.5', marker='.', alpha=0.5)
        ax.scatter([0], [0], [0], c='r', marker='.', s=10, alpha=0.5)
        minmax_list = show_cs(np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), np.array([0., 0., 0.]), ax)
        for ind in xrange(3):
            coub_axis[ind] += [minmax_list[0][ind], minmax_list[1][ind]]
        minmax_list = show_cs(cmr_rmtx.T, cmr_tvec.T[0], ax)
        for ind in xrange(3):
            coub_axis[ind] += [minmax_list[0][ind], minmax_list[1][ind]]


        import test_contours as tc
        #contours_list_3d_model_cs = list()
        for contour in tc.test_contour:
            contour_3d_model_cs = pr2t3.convert_drawing_pnts_2d_3d(np.array(contour, dtype='float64'), map(float, tc.shape), (sp.paper_sheet_vertexes[1, 0], sp.paper_sheet_vertexes[2, 1]))
            #print contour_3d_model_cs
            contour_3d_cmr_cs, minmax_list = pr2t3.sheet_crds_to_cmr_crds(cmr_rvec, cmr_tvec, contour_3d_model_cs)
            #print contour_3d_cmr_cs
            contour_3d_cmr_cs = contour_3d_cmr_cs.T
            ax.plot(contour_3d_cmr_cs[0], contour_3d_cmr_cs[1], contour_3d_cmr_cs[2], c='0.5', marker='.', alpha=0.5)

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
    pass

if __name__ == "__main__":
    #vc = cv2.VideoCapture(0)
    PORT = 9559
    IP = "192.168.1.35"
    import retrieve_bottom_cmr_frame as ret_img
    img_client = ret_img.NAOcam(IP, PORT)
    test_visualise(img_client)
    pass
