import os 
path_name = "/home/guests/demir_arikan/comp_surg/filtered"
output = []
for patient in os.listdir(path_name):
    if os.path.isdir(os.path.join(path_name, patient)):
            for image in os.listdir(os.path.join(path_name, patient, '2D')):
                    if image.endswith('.png'):
                           output.append(os.path.join(path_name, patient, '2D', image ))
                            
print(output)
print(len(output))


