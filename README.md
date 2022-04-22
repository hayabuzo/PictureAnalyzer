# PictureAnalyzer (python 3, skimage, numpy)
Script for analyzing and sorting image files by color, brightness, details.

![image](https://user-images.githubusercontent.com/38255514/164472639-6b7b4f7f-3d9f-4615-884a-bb55324399b5.png)

This script renames image files according to their:
1. Brightness (amount and balance between left and right part of image).
2. Detaliling (amount and balance between left and right part of image).
3. Saturation (amount and balance between left and right part of image).
4. Color tone, Hue (value).

Script allows to sort and filter out images with unwanted values of the parameters above (with too much or too little brightness, details, balance etc).

You can turn on the visualisation ("preview" in line 203) to see the measurements of image:

![image](https://user-images.githubusercontent.com/38255514/164474626-5be378de-591a-46d1-a399-0035161c5f63.png)

Above is the balance and below is the quantity of:
* **CYAN** - total balance
* **WHITE** - brightness
* **GREEN** - details
* **RED** - saturation
