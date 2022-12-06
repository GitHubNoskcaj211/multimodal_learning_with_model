import os

transcriptions_file_path = 'D:/JIGSAWS/Suturing/transcriptions/'
transcrptions_folder = os.listdir(transcriptions_file_path)
kinematics_file_path = 'D:/JIGSAWS/Suturing/kinematics/AllGestures/'
kinematics_folder = os.listdir(kinematics_file_path)
output_kinematics_file_path = 'D:/JIGSAWS/Suturing/kinematics/AllGesturesWithLabels/'

assert len(transcrptions_folder) == len(kinematics_folder)

for i in range(len(kinematics_folder)):
    output = open(output_kinematics_file_path + transcrptions_folder[i], "w")
    kinematics = open(kinematics_file_path + kinematics_folder[i], "r")
    gestures = open(transcriptions_file_path + transcrptions_folder[i], "r").readlines()
    current_gesture = "UNKNOWN"
    current_line = -1
    next_line = 0
    for ii, line in enumerate(kinematics):
        if current_line != -1:
            if int(gestures[current_line].split(' ')[1]) < ii:
                current_gesture = "UNKNOWN"
                current_line = -1
        if current_line == -1:
            if next_line < len(gestures) and len(gestures[next_line].split(' ')) > 3 and int(gestures[next_line].split(' ')[0]) <= ii:
                current_gesture = gestures[next_line].split(' ')[2]
                current_line = next_line
                next_line += 1
            
        output.write(f'{current_gesture}\n')
    output.close()
    kinematics.close()