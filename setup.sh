#!/bin/bash

#first install the malmo package
pip3 install malmo
cd ~
python3 -c 'import malmo.minecraftbootstrap; malmo.minecraftbootstrap.download()'


malmo_folder=MalmoPlatform
env_variable="export MALMO_XSD_PATH=/home/$USER/$malmo_folder/Schemas"
#mc_world_name="flowers_v4"

#Set environment variable in bashrc
if ! grep -q "$env_variable" ~/.bashrc; then
  echo $env_variable >> ~/.bashrc
fi
if ! grep -q "export MALMO_DIR=$malmo_folder" ~/.bashrc; then
  echo "export MALMO_DIR=$malmo_folder" >> ~/.bashrc
fi

#Copy world in Minecraft save folder
#if [ ! -d "~/$malmo_folder/Minecraft/run/saves/$mc_world_name" ]; then
#  # Control will enter here if $DIRECTORY doesn't exist.
#  cp -R $mc_world_name/ ~/$malmo_folder/Minecraft/run/saves/
#fi

echo $malmo_folder > minecraft_version_config.txt

#Copy Malmo lib
#cp -R ~/$malmo_folder/Python_Examples/MalmoPython.so gym2/envs/malmo/
#cp -R ~/$malmo_folder/Python_Examples/MalmoPython.so .

#Copy Minecraft optimized configuration
cp options.txt ~/$malmo_folder/Minecraft/run/
echo "You can now source your bashrc"