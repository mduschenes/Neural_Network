# Create Folder with Files to Download

readonly folder=/tSNE_Run_Script/dataset/tsne/*.pdf

readonly remote_dir=/work/mduschen$folder

#dataset/test_Tsweep_qLN10e5/

readonly local_dir=./dataset/tsne/dataset_May21_12pm/scharcnet

mkdir -p $local_dir

#""C:\Users\Matt\Google Drive\PSI\PSI Essay\PSI Essay Python Code\tSNE_Potts\dataset\""
# ""c/users/matt/google\ drive/psi/psi\ essay/psi\ essay\ python\ code/tSNE_Potts/dataset""
# Language Scripts
# cp *.py* $folder

# cp *.c $folder

# cp -r build $folder

# Bash Scripts
# cp command_script.sh $folder

# cp args_file.txt $folder

# Secure Copy Folder
scp -r mduschen@orca.sharcnet.ca:$remote_dir $local_dir

