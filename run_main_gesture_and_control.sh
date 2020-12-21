#export DISPLAY=:0.0
xhost local:root

#sudo docker run --gpus '"device=0"' --network="host" --env DISPLAY=$DISPLAY --env="QT_X11_NO_MITSHM=1" -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v /dev/shm:/dev/shm -v ~/Desktop/abr-demo_gesture:/home/app -it --rm --name Gesture_container woongjae94/abr-demo:gesture

sudo docker run --device="/dev/video0:/dev/video0" --gpus '"device=0"' --env DISPLAY=$DISPLAY --env="QT_X11_NO_MITSHM=1" -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v /dev/shm:/dev/shm -v ~/Desktop/abr-demo-gesture_for_eval:/home/app -it --rm --name Gesture_container woongjae94/abr-demo:gesture python main_gesture_and_control_for_test.py
