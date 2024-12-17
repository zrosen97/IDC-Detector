1) Run "./download_histopathology.sh"
2) Change line 13 in IDC_Detector.R to "image_dir <- YOUR_DATASET_DIRECTORY_HERE" where the directory is the same as the one specified in download_histopathology.sh
3) Make sure you have all of the R packages installed
4) Run IDC_Detector.R
5) That's it!  You can also use plot(index) to plot test images given a specific index (also shows CNN prediction and actual diagnosis)
