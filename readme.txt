# Final Project - Video Stabilization

My GT Box for this project:
https://gatech.box.com/s/h97a2tm7d2ycz19plw9wsedqmkan8fy7
- Readme.txt
- Input 1, Input 2, and Input 3 (original shaky videos)
- Result 1, Result 2, and Result 3 (stabilized videos, motion plots, and frames)
- Skater original, and Skater stabilized videos

Requirements: 
os
numpy
opencv
glob
matplotlib
cvxpy

1. Source code: final_project.py and stabilization.py

2. Place video in the same directory as the source code

3. Setup environment 
   PyCharm IDE setup instruction on MAC:
        1. Open final_project.py and change input filename at the VIDEO variable
        2. Open PyCharm > Preferences
        3. Expand Project: [project name] > Python Interpreter
        4. Select [+]  to install available packages
        5. Search: os> Install Package
        6. Search: numpy > Install Package
        7. Search: opencv-python3 > Install Package
        8. Search: glob > Install Package
        9. Search: matplotlib > Install Package
	10. Search: cvxpy > Install Package
        11. Click OK
        12. Run final_project.py
    Or install libraries and run python using command line terminal:
	pip install os
	pip install numpy
        pip install opencv-python
        pip install glob
        pip install matplotlib
	pip install cvxpy
        python3 final_project.py
	* Need to open final_project.py and update the VIDEO variable first before running this file

4. The output videos will be in \output\<video name>\rect\result.mp4
   The output videos will be in \output\<video name>\crop\result.mp4

   The video in the rect will have a center rectangle in the middle.
   The video in the crop directory will not have any black outline.
