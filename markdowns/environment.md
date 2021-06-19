# Environment Preparation

## Overview: 
I've used a mixed Linux/Windows environment, using WSL to have Ubuntu on a Windows 10 OS. I installed all the software in Linux except Visual Studio, for personal preference.

 Tools used:

 - Anaconda for Python environments (Linux/Windows)
 - Tensorflow 2 and Jupyter notebook (Linux/Windows)
 - Dotnet SDK (Linux/Windows), or any C# compiler
 - Clang++9 (Linux). This also will be used to export to CG.
  - Visual Studio for C++ coding (Windows /Optional)

You can do all the training pipeline in just one OS, either Windows or linux,  sometimes linux binaries are faster than Visual Studio counterpart, measure it.
To export the bot to Codingame you need the CGZero binary in x64 Linux, you can't send a Windows binary.  If you strip/codegolf the code maybe it's feasible to send the source .cpp code + weights file as a string in the code. The code has a function to convert float32 weights to float16, that reduces weight file to 50%, with a low accuracy error. So doing that + some compression to Unicode it can probably fit.

**WARNING! Sending compiled binaries are forbidden on Challenges! Don't use them because you can be disqualified. It's OK tu use it on multi games.**

## Steps:

 1. Install Anaconda
https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/

2. Create a Tensorflow environment, including Python, Tensorflow, Numpy and Jupyter Notebook
``` bash
conda create -n tf tensorflow matplotlib numpy
conda activate tf
conda install -c conda-forge notebook
```
For GPU acceleration on tensorflow you'll need additional libraries (check https://www.tensorflow.org/install/gpu ). This is out of scope.

3. If you use DotNet SDK you'll need a separate folder to create a project for `ENCODER16k.cs`. Create the folder, then `dotnet new console` inside it, and paste the .cs file there. With mono or other SDK's you can just build a single file "binary"
4. Create a folder for the Training pipeline. Copy all files (except `ENCODER16k.cs` if you used DotNet SDK):

**image**

5. Compile `NNSampler.cpp` and `CGZero.cpp` binaries. On Linux I use:


``` bash
clang++-9 -std=c++17 -march=core-avx2 -mbmi2 -mpopcnt -mavx2 -mavx -mfma -O3 -fomit-frame-pointer -finline "$AI_Name.cpp" -lpthread -o "$AI_Name"
strip -S --strip-unneeded --remove-section=.note.gnu.gold-version --remove-section=.comment --remove-section=.note --remove-section=.note.gnu.build-id --remove-section=.note.ABI-tag "$AI_Name"
upx "$AI_Name" -9 --best --ultra-brute --no-backup --force
```
In Visual Studio, Release compilation parameters are:
``` C++
/JMC /permissive- /GS /W3 /Zc:wchar_t /ZI /Gm- /Od /sdl /Fd"x64\Debug\vc141.pdb" /Zc:inline /fp:precise /D "_DEBUG" /D "_CONSOLE" /D "_CRT_SECURE_NO_WARNINGS" /D "_UNICODE" /D "UNICODE" /errorReport:prompt /WX- /Zc:forScope /RTC1 /arch:AVX2 /Gd /MDd /std:c++17 /FC /Fa"x64\Debug\" /EHsc /nologo /Fo"x64\Debug\" /Fp"x64\Debug\CGZero.pch" /diagnostics:classic 
```
It needs `AVX2`, `C++17`

**NOTE: You need to have defined your NN Model before compiling  `CGZero.cpp`** , it will fail to compile as it lacks the Create NN Model function.

6. Start Jupyter Notebook.
```
conda activate tf (only if you aren't in the tf virtual environment) 
jupyter notebook
```
This will open a browser, if it's not open (like in mixed Windows/Linux environments) you'll see something like:
```
[I 13:57:41.110 NotebookApp] Serving notebooks from local directory: /mnt/e/folder
[I 13:57:41.110 NotebookApp] The Jupyter Notebook is running at:
[I 13:57:41.111 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 13:57:41.470 NotebookApp]
    To access the notebook, open this file in a browser:
        http://localhost:8889/?token=a5e120ecc047a4d6381fcbd2
```
Copy the **`http://localhost:8889/?token=<thisisatoken>`** URL on a webbrowser, and you'll access Jupyter Notebook. Open `Train.ipynb` file from the browser.

7. Click the second cell, and press `Run`, if Tensorflow is correctly installed you'll see:
```
Python:3.7.4 (default, Aug 13 2019, 20:35:49) 
[GCC 7.3.0]
TF:2.4.1
WARNING:tensorflow:From <ipython-input-1-f0240f82491d>:24: is_gpu_available (from tensorflow.python.framework.test_util) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.config.list_physical_devices('GPU')` instead.
GPU:False CUDA:True
```
The warning is for a deprecated call to check if GPU is active, it's irrelevant.

8. The next section is the Hyperparameters section, configure values to change training behavior. I can't recommend anything because it's obscure to me, I did trial and error. `THREADS` is important to use multithreading, set up accordly to your CPU.

9. Go to the cell with the MODEL DEFINITION. Define your neural network model. Run to compile the model. **This NN Model must match exactly what you have in the .cpp code.**

10. Near the end of the script, in auxiliary tools, there is a script for Model Validation between C++ and tensorflow. **Ensure that predicted values are the same. If you don't have this point you'll never be able to train anything.**

11. Once you have a correctly installed Tensorflow, your hyperparameters for training, your Model loaded and synchronized between Tensorflow and C++ binary, you are ready to train. Just go to first cell and press Run on all cells until you reach the **MAIN TRAINING LOOP**. At this point the code will run indefinitely until you press stop. It should be safe to stop at any time, but I prefer to cancel it when it's generating selfplays. If you stop on tensorflow training or pit winrate calculation the generation won't get a score. It shouldn't be a problem but it's undesired. Self-play is a safe point to stop, it won't break anything.

## Steps for resume training:
1. On a terminal/console, go to the folder with all the code.
2. Start Jupyter Notebook, connect to the url.
``` bash
conda activate tf
jupyter notebook
```
3. In the browser, open `Train.ipynb` file.
4. Click run until you reach the **MAIN TRAINING LOOP**.
