## Custom Dockerfile allowing remote display of Malmo installed using the pip3 Python wheel
#FROM consol/ubuntu-xfce-vnc
FROM ubuntu:16.04
ENV REFRESHED_AT 2018-03-18

## Install as root 
USER 0

# 16.04 image doesn't contain sudo - install that first:
RUN apt-get update && apt-get install -y sudo

# Create a user called "malmo", give it sudo access and remove the requirement for a password:
RUN useradd --create-home --shell /bin/bash --no-log-init --groups sudo malmo
RUN sudo bash -c 'echo "malmo ALL=(ALL:ALL) NOPASSWD: ALL" | (EDITOR="tee -a" visudo)'

# Fix weird keyboard layout question when installing xpra
RUN sudo bash -c 'echo "XKBLAYOUT=\"xx\"" >> /etc/default/keyboard'

# While we are still root, install the necessary dependencies for Malmo:
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    openjdk-8-jdk \
    ffmpeg \
    software-properties-common \
    xpra \
    libgl1-mesa-dri

RUN sudo update-ca-certificates -f
RUN sudo pip3 install numpy matplotlib scipy

# Note the trailing slash - essential!
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
RUN echo "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/" >> /home/malmo/.bashrc

# Switch to the malmo user:
USER malmo
WORKDIR /home/malmo

# Set MALMO_XSD_PATH to download install location.
ENV MALMO_XSD_PATH=/home/malmo/MalmoPlatform/Schemas

RUN sudo pip3 install malmo
RUN python3 -c "import malmo.minecraftbootstrap;malmo.minecraftbootstrap.download(buildMod=True)"

# precompile stuff
RUN echo 'org.gradle.daemon=true\n' > ~/.gradle/gradle.properties
WORKDIR /home/malmo/MalmoPlatform/Minecraft
RUN ./gradlew setupDecompWorkspace
RUN ./gradlew build

COPY options.txt /home/malmo/MalmoPlatform/Minecraft/run/
RUN sudo apt-get install -y dos2unix
COPY ./console_startup.sh /home/malmo/console_startup.sh
RUN sudo dos2unix /home/malmo/console_startup.sh
ENTRYPOINT ["/home/malmo/console_startup.sh"]
CMD []
