# Create Folder with Files to Upload

readonly folder=./tSNE_dataset_May21_6pm

mkdir -p $folder/dataset/tsne/

# Language Scripts
cp *.py* $folder

#rm $folder/*.pyd

#cp *.c $folder

#cp -r build $folder

# Bash Scripts
#cp command_script.sh $folder

cp args_file.txt $folder

cp -r ./dataset/tsne/Ising_Gauge_Data/*.npz $folder/dataset/tsne/

# Secure Copy Folder
scp -r $folder mduschen@orca.sharcnet.ca:/work/mduschen/

# Delete Directory
rm -r $folder/