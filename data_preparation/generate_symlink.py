import os

source_base_folder = '/data3/cj/Obj_Data/speed_user_study_datasets/self'  # Path to the base folder containing the source folders
target_base_folder = '/data3/cj/Obj_Data/speed_user_study_datasets/all/dataset'  # Path to the target base folder where symlinks will be created

participants = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P10', 'P11', 'P12', 'P13']  # List of participant folders. Will be replaced to p1, p2... for easier usage
# participants = ['1.5', '2.0', '2.5']

# sessions=[i for i in range(1,22)]
sessions=['01','02','03','04','05','06','07']

# x=0

num = len(participants)
for participant in participants:
    # dataset_root = participants[p - 1]
    source_folder = os.path.join(source_base_folder, participant, 'dataset') 
    target_folder = os.path.join(target_base_folder) 

    os.makedirs(target_folder, exist_ok=True)

    for data_subfolder in os.listdir(source_folder):
        session_name=data_subfolder[-4:-2]
        # print(session_name)
        # x=x+1
        # print(x)
        # print(data_subfolder[-4:])

        if session_name in sessions:

            source_session_folder=os.path.join(source_folder,data_subfolder)
            target_session_folder=os.path.join(target_folder,'session_{}_{}'.format(participant, data_subfolder[-7:]))

            # os.makedirs(target_session_folder, exist_ok=True)
            # session_folder = 'session_{:02d}'.format(session)  
            # source_session_folder = os.path.join(source_folder, session_folder)  
            # target_session_folder = os.path.join(target_folder, 'session_{}_{}'.format(participant, session_folder)) 

            print(source_session_folder)
            print(target_session_folder)

            os.symlink(source_session_folder, target_session_folder)

print("Symlinks created successfully!")