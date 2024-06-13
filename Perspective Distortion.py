import cv2 as cv
import numpy as np
alphabet=["a","b","c","d"]
# Load the image
original = cv.imread("card.png", 1)
cv.imshow("original picture",original)
gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)

# Apply thresholding and morphological operations
_, img = cv.threshold(gray, 240, 255, cv.THRESH_BINARY_INV)
img = cv.morphologyEx(img, cv.MORPH_CLOSE, (3,3), iterations=9)

# Shi-Tomasi corner detection
corners = cv.goodFeaturesToTrack(img, 100, 0.1, 100)

# Check if at least 4 corners are detected
if corners is not None and len(corners) >= 4:
    corners = np.intp(corners)

    # Select the first 4 corners
    src = np.array(corners[:4], dtype=np.float32).reshape(-1, 2)

    #Define the destination points
    dst = np.array(([0,300],[0,0],[200,0],[200,300]), dtype=np.float32)

    #Get the perspective transform matrix
    matrix = cv.getPerspectiveTransform(src, dst)

    dst=cv.warpPerspective(original,matrix,img.shape)
    cv.imshow("rotated",dst)

    # Draw the detected corners on the original image
    a=0
    for corner in corners[:4]:
        x, y = corner.ravel()
        cv.circle(original, (x, y), 5, (0, 255, 0), -1)
        cv.putText(original,str(f"{alphabet[a]}({x},{y})"),(x,y),cv.FONT_HERSHEY_PLAIN,1,(0,0,255))
        a=a+1
    # Display the images
    cv.imshow("dots", original)
    cv.imshow("img", img)

    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("Not enough corners detected.")
