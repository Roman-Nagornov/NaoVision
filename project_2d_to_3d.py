import numpy as np
import cv2


def convert_drawing_pnts_2d_3d(in_drawing_pnts, in_drawing_size, in_paper_sheet_size):
    """
        Function for projecting of image pxls points into 3D paper sheet with
        known coordinates of vertexes.
        Input:
            in_drawing_pnts - (Nx2 numpy array) array of the image points,
                in imagei's pxls coordinate system, that we want to draw
            in_drawing_size - ([height, width] numpy array of size 2)
                the size of image in pxls.
            in_papersheet_vertexes - (4x3 numpy array) array of 3D coordinates
                of paper sheet in space.
        Output:
            out_drawing_3d_pnts - (Nx3 numpy array) array of 2D image pxl coordinates
                projected into 3D paper sheet.
    """
    drawing_pnts = in_drawing_pnts.copy()
    # Number of image points to be projected.
    drawing_pnts_num = drawing_pnts.shape[0]
    print drawing_pnts
    # Convert all pxl image point coordinates into relative fractions.
    drawing_pnts[:, 0] = drawing_pnts[:, 0] / float(in_drawing_size[0]) * float(in_paper_sheet_size[0])
    drawing_pnts[:, 1] = drawing_pnts[:, 1] / float(in_drawing_size[1]) * float(in_paper_sheet_size[1])
    print drawing_pnts
    zeros_pad = np.zeros((drawing_pnts_num, 1))
    drawing_pnts_3d = np.concatenate((drawing_pnts, zeros_pad), axis=1)

    return drawing_pnts_3d


def project_2d_to_3d(in_drawing_pnts, in_drawing_size, in_papersheet_vertexes):
    """
        Function for projecting of image pxls points into 3D paper sheet with
        known coordinates of vertexes.
        Input:
            in_drawing_pnts - (Nx2 numpy array) array of the image points,
                in imagei's pxls coordinate system, that we want to draw
            in_drawing_size - ([height, width] numpy array of size 2)
                the size of image in pxls.
            in_papersheet_vertexes - (4x3 numpy array) array of 3D coordinates
                of paper sheet in space.
        Output:
            out_drawing_3d_pnts - (Nx3 numpy array) array of 2D image pxl coordinates
                projected into 3D paper sheet.
    """
    drawing_pnts = in_drawing_pnts.copy()
    # Number of image points to be projected.
    drawing_pnts_num = drawing_pnts.shape[0]

    # Convert all pxl image point coordinates into relative fractions.
    drawing_pnts[:, 0] /= in_drawing_size[0]
    drawing_pnts[:, 1] /= in_drawing_size[1]

    # Find the projected point on 3D paper sheet with relative fractions.
    out_drawing_3d_pnts = np.zeroes(drawing_pnts_num, 3)
    top_x_axis_vect = in_papersheet_vertexes[1] - in_papersheet_vertexes[0]
    bottom_x_axis_vect = in_papersheet_vertexes[3] - in_papersheet_vertexes[2]
    for ind in xrange(drawing_pnts_num):
        top_x_axis_pnt = in_papersheet_vertexes[0] + top_x_axis_vect * drawing_pnts[ind, 0]
        bottom_x_axis_pnt = in_papersheet_vertexes[4] + bottom_x_axis_vect * drawing_pnts[ind, 0]
        out_drawing_3d_pnts[ind] = top_x_axis_pnt + (bottom_x_axis_pnt - top_x_axis_pnt) * drawing_pnts[ind, 1]
    return out_drawing_3d_pnts

def sheet_crds_to_cmr_crds(rvec, tvec, dots_list_sheet_cs):
    rmtx, jacobian = cv2.Rodrigues(rvec)
    dots_list_cmr_cs = (np.dot(rmtx, dots_list_sheet_cs.T) + tvec)
    minmax_list = [np.amin(dots_list_cmr_cs, axis=0), np.amax(dots_list_cmr_cs, axis=0)]
    return dots_list_cmr_cs, minmax_list

if __name__ == "__main__":
    import ShowProjection as sp
    import cv2
    drawing = cv2.imread('test.png')
    is_detected, paper_sheet_pnts, cmr_rvec, cmr_tvec = sp.detect_paper_sheet(drawing)
    print "original coord: ",
    pass
