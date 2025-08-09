git clone https://github.com/BesomiE/my-files.git

cd my-files.git
chmod +x setup.sh
./setup.sh

❌ Issue:
You were getting the error:

Camera(s) not found (Do not forget to disable legacy camera with raspi-config).

Also:

libcamera-hello
→ ERROR: *** no cameras available ***

🔍 Cause:

The Raspberry Pi camera (using Picamera2 and libcamera) wasn't being detected.

You were using the correct arm_64bit=1, and legacy camera support was already disabled.

However, /boot/config.txt was missing required settings to enable modern camera stack.

⚙️ Solution:

-Edit /boot/config.txt:

sudo nano /boot/config.txt


-Scroll and make sure these lines exist and are set correctly:
Ensure these lines are present:

start_x=1
gpu_mem=128
camera_auto_detect=1
dtoverlay=vc4-kms-v3d

 If you see start_x=0 or camera_auto_detect=0, change them to 1.
 
-If you're using the new libcamera stack (not the old legacy camera), also comment out any legacy overlay:
Make sure this line is NOT present (or is commented out):

#dtoverlay=vc4-fkms-v3d

-Save and exit:

Press Ctrl+O → Enter to save

Press Ctrl+X to exit

-reboot:

sudo reboot

After reboot, test the camera:

libcamera-hello
