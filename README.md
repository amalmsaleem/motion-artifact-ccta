# Cycle Consistent Generative Motion Artifact Correction in Coronary Computed Tomography Angiography

This repository contains the code for the preprocessing and image reconstruction method described in the paper: [Cycle Consistent Generative Motion Artifact Correction in Coronary Computed Tomography Angiography](https://www.mdpi.com/2076-3417/14/5/1859)

The implementation uses a cycle-consistent generative model to reconstruct coronary CT images by correcting motion artifacts, improving image quality for better clinical interpretation.

## Features
- Extraction of motion artifact-affected coronary CT images, as shown below.

 ![image](https://github.com/user-attachments/assets/7fc70b71-c200-4e65-9df0-afe2565d23de)

 
- Preprocessing of extracted patches.
- Reinsertion of said patches to their exact locations.
