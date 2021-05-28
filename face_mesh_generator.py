import cv2 
import mediapipe as mp  


img = cv2.imread('ad.jpg') #put your image here  

mp_dace_mesh = mp.solutions.face_mesh.FaceMesh()     


rgb_image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 


resultat = mp_dace_mesh.process(rgb_image)


h, w, _ = img.shape

for facial_landmarks in resultat.multi_face_landmarks : 
    for i in range(468) :
        pt1 = facial_landmarks.landmark[i]
        print(pt1) 
        x = int(pt1.x * w) 
        y = int(pt1.y * h)
        cv2.circle(img, (x,y),2,(0,100,100),-1)



cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
