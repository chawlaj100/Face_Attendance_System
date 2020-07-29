import dlib
import numpy as np
import cv2         
import os        
import shutil     
# Dlib frontal face detector
detector = dlib.get_frontal_face_detector()

# OpenCv Uses camera
cap = cv2.VideoCapture(1)

# The counter for screen-shot
cnt_ss = 0

# The folder to save face images
current_face_dir = ""

# The directory to save images of faces
path_photos_from_camera = "data/data_faces_from_camera/"


# Make directory for saving photos and csv
def pre_work_mkdir():

    # Make folders to save faces images and csv
    if os.path.isdir(path_photos_from_camera):
        pass
    else:
        os.mkdir(path_photos_from_camera)


pre_work_mkdir()

# Delete the old data
def pre_work_del_old_face_folders():
    folders_rd = os.listdir(path_photos_from_camera)
    for i in range(len(folders_rd)):
        shutil.rmtree(path_photos_from_camera+folders_rd[i])

    if os.path.isfile("data/features_all.csv"):
        os.remove("data/features_all.csv")

# Uncomment the below line to delete the old data in directories person_1, person_2, person_3...
# pre_work_del_old_face_folders()
##################################


# Checking order of people: person_cnt
# If the old folders exists
if os.listdir("data/data_faces_from_camera/"):
    # Get the num of latest person
    person_list = os.listdir("data/data_faces_from_camera/")
    person_num_list = []
    # for person in person_list:
    #     person_num_list.append(int(person.split('_')[-1]))
    # person_cnt = max(person_num_list)
    # person_cnt = person_list.count

#  flag / The flag to control if save
save_flag = 1

# 'n'  's' The flag to check whether 'n' is pressed before 's'
press_n_flag = 0

while cap.isOpened():
    flag, img_rd = cap.read()
    # print(img_rd.shape)
    # It should be 480 height * 640 width

    kk = cv2.waitKey(1)

    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
    
    # Faces
    faces = detector(img_gray, 0)

    # Font
    font = cv2.FONT_ITALIC

    # press 'n' to create the folders for saving faces
    if kk == ord('n'):
        # person_cnt += 1
        while True:
            print("Enter Folder Name")
            new_dir_name = input()

            current_face_dir = path_photos_from_camera + new_dir_name
            if os.path.isdir(current_face_dir) == False:
                break
            else:
                print("Already Exists")
        os.makedirs(current_face_dir)
        print('\n')
        print("Create folders: ", current_face_dir)

        cnt_ss = 0              #  clear the count of faces
        press_n_flag = 1        #  'n' have pressed 'n'

    # Face detected ?
    if len(faces) != 0:
        # Show the rectangle box of face
        for k, d in enumerate(faces):
            # Compute the width and height of the box
            # (x,y) 
            pos_start = tuple([d.left(), d.top()])
            pos_end = tuple([d.right(), d.bottom()])

            # compute the size of rectangle box
            height = (d.bottom() - d.top())
            width = (d.right() - d.left())

            hh = int(height/2)
            ww = int(width/2)

            # the color of rectangle box
            color_rectangle = (255, 255, 255)

            # 480x640
            if (d.right()+ww) > 640 or (d.bottom()+hh > 480) or (d.left()-ww < 0) or (d.top()-hh < 0):
                cv2.putText(img_rd, "OUT OF RANGE", (20, 300), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                color_rectangle = (0, 0, 255)
                save_flag = 0
                if kk == ord('s'):
                    print("Please adjust your position")
            else:
                color_rectangle = (255, 255, 255)
                save_flag = 1

            cv2.rectangle(img_rd,
                          tuple([d.left() - ww, d.top() - hh]),
                          tuple([d.right() + ww, d.bottom() + hh]),
                          color_rectangle, 2)

            #  Create blank image according to the shape of face detected
            im_blank = np.zeros((int(height*2), width*2, 3), np.uint8)

            if save_flag:
                # 5. Press 's' to save faces into images
                if kk == ord('s'):
                    # check if you have pressed 'n'
                    if press_n_flag:
                        cnt_ss += 1
                        for ii in range(height*2):
                            for jj in range(width*2):
                                im_blank[ii][jj] = img_rd[d.top()-hh + ii][d.left()-ww + jj]
                        cv2.imwrite(current_face_dir + "/img_face_" + str(cnt_ss) + ".jpg", im_blank)
                        print("Save into: ", str(current_face_dir) + "/img_face_" + str(cnt_ss) + ".jpg")
                    else:
                        print("Please press 'N' before 'S'")

    # Show the numbers of faces detected
    cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    # Add some statements
    cv2.putText(img_rd, "Face Register", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img_rd, "N: Create face folder", (20, 350), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img_rd, "S: Save current face", (20, 400), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img_rd, "Q: Quit", (20, 450), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

    # 6. Press 'q' to exit
    if kk == ord('q'):
        break

    cv2.imshow("camera", img_rd)

# Release camera and destroy all windows
cap.release()
cv2.destroyAllWindows()
