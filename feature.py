
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.feature import hog
from skimage import data, exposure
class Feature:
    def __init__(self, path):
        self.path=path 
    def hog(self):
        image = cv2.imread(self.path)

        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off') 
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()

    def sift(self):
        import numpy as np
        import cv2 
        img = cv2.imread(self.path)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp = sift.detect(gray,None)

        img=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite('sift_keypoints.jpg',img)
        kp,des = sift.compute(gray,kp)

    def lbp(self):
        def get_pixel(img, center, x, y):
            
            new_value = 0
            
            try:
                # If local neighbourhood pixel value is greater than or equal to center pixel values then set it to 1
                if img[x][y] >= center:
                    new_value = 1
                    
            except:
                # Exception is required when neighbourhood value of a center pixel value is null i.e. values present at boundaries.
                pass
            
            return new_value

        # Function for calculating LBP
        def lbp_calculated_pixel(img, x, y):

            center = img[x][y]

            val_ar = []
            
            # top_left
            val_ar.append(get_pixel(img, center, x-1, y-1))
            
            # top
            val_ar.append(get_pixel(img, center, x-1, y))
            
            # top_right
            val_ar.append(get_pixel(img, center, x-1, y + 1))
            
            # right
            val_ar.append(get_pixel(img, center, x, y + 1))
            
            # bottom_right
            val_ar.append(get_pixel(img, center, x + 1, y + 1))
            
            # bottom
            val_ar.append(get_pixel(img, center, x + 1, y))
            
            # bottom_left
            val_ar.append(get_pixel(img, center, x + 1, y-1))
            
            # left
            val_ar.append(get_pixel(img, center, x, y-1))
            
            # Now, we need to convert binary
            # values to decimal
            power_val = [1, 2, 4, 8, 16, 32, 64, 128]

            val = 0
            
            for i in range(len(val_ar)):
                val += val_ar[i] * power_val[i]
                
            return val

        img_bgr = cv2.imread(self.path, 1)

        height, width, _ = img_bgr.shape

        # We need to convert RGB image into gray one because gray image has one channel only.
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Create a numpy array as the same height and width of RGB image
        img_lbp = np.zeros((height, width),np.uint8)

        for i in range(0, height):
            for j in range(0, width):
                img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)

        plt.imshow(img_bgr)
        plt.show()

        plt.imshow(img_lbp, cmap ="gray")
        plt.show()

        print("LBP Program is finished")

f1= Feature('F:\periocular recognition\hog\im_patch.jpg')
f1.hog()
f1.sift()
f1.lbp()

