import cv2
import numpy as np

#----------------------------------------------------------------------#
#                           flattener function
#----------------------------------------------------------------------#


#Flattens an image of a card into a top-down 200x300 perspective.
def flattener(image, pts, w, h):
    
    temp_rect = np.zeros((4,2), dtype = "float32")
    
    s = np.sum(pts, axis = 2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # An array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.8*h: # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # If the card is 'diamond' oriented, a different algorithm
    # has to be used to identify which point is top left, top right
    # bottom left, and bottom right.
    
    if w > 0.8*h and w < 1.2*h: #If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0] # Top left
            temp_rect[1] = pts[0][0] # Top right
            temp_rect[2] = pts[3][0] # Bottom right
            temp_rect[3] = pts[2][0] # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0] # Top left
            temp_rect[1] = pts[3][0] # Top right
            temp_rect[2] = pts[2][0] # Bottom right
            temp_rect[3] = pts[1][0] # Bottom left
            
        
    maxWidth = 200
    maxHeight = 300

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)

    return warp

#---------------------------------  End  -----------------------------------#


#---------------------------------------------------------------------------#
#                   Initialization and Pre-processing of images
#---------------------------------------------------------------------------#
# Initialize image
image = cv2.imread('t.png')

# For template size purpose
tempsize = cv2.imread('diamond.png',0)

# Initializing array of template images
heartpic = cv2.imread('heart.png',0)
spadepic =cv2.imread('spade.png',0)
diamondpic = cv2.imread('diamond.png',0) 
clubpic = cv2.imread('club.png',0)
template = [diamondpic, heartpic, spadepic, clubpic]

# Pre-process image
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
retval, thresh = cv2.threshold(blur,100,255,cv2.THRESH_BINARY)

# Find contours and sort them by size
_,cnts,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea,reverse=True)

# Draw contours
cv2.drawContours(image, cnts, -1, (0,255,0), 3)
cv2.imshow("cnts", image)
cv2.waitKey()
cv2.destroyAllWindows()

# If there are no contours, print an error
if len(cnts) == 0:
    print('No contours found!')
    quit()

# Declaring suit count and size of template
hearts = 0
spades = 0
clubs = 0
diamonds = 0
w2, h2 = tempsize.shape[::-1]

#-----------------------------------  End  --------------------------------------#


#--------------------------------------------------------------------------------#
#                               Main Loop
#--------------------------------------------------------------------------------#

#Loop through each detected contours
for i in range(len(cnts)):
    # Approximate the corner points of the card
    peri = cv2.arcLength(cnts[i],True)
    approx = cv2.approxPolyDP(cnts[i],0.01*peri,True)
    pts = np.float32(approx)       
    x,y,w,h = cv2.boundingRect(cnts[i])
    
#    cv2.rectangle(image, (x,y), (x + w, y + h), (0,0,255), 2)
#    cv2.imshow("cnts", image)
    
    # Flatten the card and convert it to 200x300
    warp = flattener(image,pts,w,h)
    cv2.imshow("warp", warp)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # Grab corner of card image
    corner = warp[0:84, 0:32]

    # Perform template matching between card and template
    # Loop through the 4 template images
    for suit in range(len(template)):
        
        res = cv2.matchTemplate(corner,template[suit],cv2.TM_CCOEFF_NORMED)        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # only perform card counting if matching rate is equal to or more than 90%
        if max_val>=0.90:
            threshold = max_val
            loc = np.where( res >= threshold)
            
            for pt in zip(loc[1], loc[0]):
                cv2.rectangle(corner, pt, (pt[0] + w2, pt[1] + h2), [0,255,255], 2)
            
                if np.any(template[suit]==diamondpic):
                    diamonds+=1
                if np.any(template[suit]==heartpic):
                    hearts+=1
                if np.any(template[suit]==spadepic):
                    spades+=1
                if np.any(template[suit]==clubpic):
                    clubs+=1
        
#        cv2.imshow("res", corner)
#        cv2.waitKey()
#        cv2.destroyAllWindows()

#---------------------------------  End  -----------------------------------------#

print("Number of Spade: ", spades)
print('Number of Diamond: ', diamonds)
print('Number of Club: ' , clubs)
print('Number of Heart: ', hearts)
print('Total Cards: ', len(cnts) )
