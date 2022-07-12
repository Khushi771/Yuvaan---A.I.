import sign_detector

import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened(): 
    success , img = cap.read()
    if(success):
        # trained cascade classifier for each object to detect
        left_cascade = cv2.CascadeClassifier('haar_trained_xml/left/cascade.xml')
        right_cascade = cv2.CascadeClassifier('haar_trained_xml/right/cascade.xml')

        left_signs = sign_detector.classify_signs(img, left_cascade)
        right_signs = sign_detector.classify_signs(img, right_cascade)

        # draw bounding box to the object detected
        sign_detector.show_box(img, left_signs)
        sign_detector.show_box(img, right_signs)

        if len(left_signs) > 0:
            print("left")
        elif len(right_signs) > 0:
            print("Right")

        cv2.imshow("Output" ,img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Error getting feed")
        break

cap.release()
cv2.destroyAllWindows()