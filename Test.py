import cv2
import os, glob

#cap = cv2.VideoCapture("Explosion001_x264.mp4")
#check if the video capture is open
#if(cap.isOpened() == False):
#    print("Error Opening Video Stream Or File")
#while(cap.isOpened()):
#    ret, frame =cap.read()
#    if ret == True:
#        cv2.imshow('frame', frame)
#        if cv2.waitKey(25)  == ord('q'):
#            break
#    else:
#        break
#cap.release()
#cv2.destroyAllWindows()
#print('Video Completed Frame Process started..');
#######################
#video_frame("Fighting002_x264.mp4")
vidcap = cv2.VideoCapture("Abuse001_x264.mp4")
success,image = vidcap.read()
count = 0
while success:
    cv2.imwrite("Database\\Train\\Abuse\\frame%d.png" % count, image)     # save frame as JPEG file
    cv2.imwrite("Database\\Test\\Abuse\\frame%d.png" % count, image)     # save frame as JPEG file
    #print("Database\\Train\\Burglary\\frame%d.png" % count, image)
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
print('Frame Process Completed..');

vidcap = cv2.VideoCapture("Arrest001_x264.mp4")
success,image = vidcap.read()
count = 0
while success:
    cv2.imwrite("Database\\Train\\Arrest\\frame%d.png" % count, image)     # save frame as JPEG file
    cv2.imwrite("Database\\Test\\Arrest\\frame%d.png" % count, image)     # save frame as JPEG file
    #print("Database\\Train\\Burglary\\frame%d.png" % count, image)
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
print('Frame Process Completed..');

vidcap = cv2.VideoCapture("Arson001_x264.mp4")
success,image = vidcap.read()
count = 0
while success:
    cv2.imwrite("Database\\Train\\Arson\\frame%d.png" % count, image)     # save frame as JPEG file
    cv2.imwrite("Database\\Test\\Arson\\frame%d.png" % count, image)     # save frame as JPEG file
    #print("Database\\Train\\Burglary\\frame%d.png" % count, image)
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
print('Frame Process Completed..');

vidcap = cv2.VideoCapture("Assault001_x264.mp4")
success,image = vidcap.read()
count = 0
while success:
    cv2.imwrite("Database\\Train\\Assault\\frame%d.png" % count, image)     # save frame as JPEG file
    cv2.imwrite("Database\\Test\\Assault\\frame%d.png" % count, image)     # save frame as JPEG file
    #print("Database\\Train\\Burglary\\frame%d.png" % count, image)
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
print('Frame Process Completed..');

vidcap = cv2.VideoCapture("Burglary001_x264.mp4")
success,image = vidcap.read()
count = 0
while success:
    cv2.imwrite("Database\\Train\\Burglary\\frame%d.png" % count, image)     # save frame as JPEG file
    cv2.imwrite("Database\\Test\\Burglary\\frame%d.png" % count, image)     # save frame as JPEG file
    #print("Database\\Train\\Burglary\\frame%d.png" % count, image)
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
print('Frame Process Completed..');

vidcap = cv2.VideoCapture("Explosion001_x264.mp4")
success,image = vidcap.read()
count = 0
while success:
    cv2.imwrite("Database\\Train\\Explosion\\frame%d.png" % count, image)     # save frame as JPEG file
    cv2.imwrite("Database\\Test\\Explosion\\frame%d.png" % count, image)     # save frame as JPEG file
    #print("Database\\Train\\Burglary\\frame%d.png" % count, image)
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
print('Frame Process Completed..');

vidcap = cv2.VideoCapture("Fighting002_x264.mp4")
success,image = vidcap.read()
count = 0
while success:
    cv2.imwrite("Database\\Train\\Fighting\\frame%d.png" % count, image)     # save frame as JPEG file
    cv2.imwrite("Database\\Test\\Fighting\\frame%d.png" % count, image)     # save frame as JPEG file
    #print("Database\\Train\\Burglary\\frame%d.png" % count, image)
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
print('Frame Process Completed..');


vidcap = cv2.VideoCapture("Normal_Videos_003_x264.mp4")
success,image = vidcap.read()
count = 0
while success:
    cv2.imwrite("Database\\Train\\Normal\\frame%d.png" % count, image)     # save frame as JPEG file
    cv2.imwrite("Database\\Test\\Normal\\frame%d.png" % count, image)     # save frame as JPEG file
    #print("Database\\Train\\Burglary\\frame%d.png" % count, image)
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
print('Frame Process Completed..');
