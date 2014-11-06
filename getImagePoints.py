import cv2
import numpy as np

def get_points(image):
    """
    Input:
        image - image that 
    """
    blured = cv2.GaussianBlur(image, (3,3), 0)
    grayscale_img = cv2.cvtColor(blured, cv2.COLOR_BGR2GRAY)
    image_vertexes = cv2.Canny(grayscale_img, 50, 150, apertureSize=3)
    all_contours, hierarchy = cv2.findContours(image_vertexes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contoured_img = np.zeros(image.shape)
    reduced_contours = all_contours
    for i in xrange(len(reduced_contours)):
        points_array =reduced_contours[i][::50]
        for point in points_array:
            cv2.circle(contoured_img, (point[0][0], point[0][1]), 1, [0, 0, 255])
    cv2.imwrite('contours.jpg', contoured_img)
    with open("out.txt", "w") as out:
        out.write(str(points_array))
    pass

def main():
    drawing = cv2.imread('test.png')
    get_points(drawing)
    pass

if __name__ == "__main__":
    main()