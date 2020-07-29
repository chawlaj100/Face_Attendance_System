# Face_Attendance_System
Marking the attendance of students/employees/workers in any organization using face recognition.

## Requirements to run the project:
1. You should have installed python>=3.5
2. Other important libs include : dlib, opencv-python, skimage, pandas, numpy, shutil, csv.

### To setup this project on your system follow thses steps:
-> Run the file 'get_faces_from_camera.py'. This will open a window where you could see you face and could perform operations like making a new folder, capturing new images, and saving the person's data. 
While saving the face data it is recommended to take atleast 5-10 images per person to get good performance in working mode. Tilt the face slightly after capturing each frame to capture all the areas.

-> Then run the file 'feature_extraction_to_csv.py' for extracting the faces from the folders containing images created in the above step.

-> And finally execute the file 'face_reco_from_camera.py' to see the performance of the project. 
