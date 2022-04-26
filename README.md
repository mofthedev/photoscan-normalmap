# photoscan-normalmap
Generating normal maps using both 'photometric stereo' and 'height map to normal map' methods

This Python code converts photometric stereo images like these:
![up](https://user-images.githubusercontent.com/84385239/165281775-6e68bd1a-7609-42c5-be1b-e96ce8cc6035.jpg)
![down](https://user-images.githubusercontent.com/84385239/165281789-e27e10ca-67b2-486f-824c-f2a06a354dbe.jpg)
![right](https://user-images.githubusercontent.com/84385239/165281860-3a4ee1a0-eac2-4fd5-bbda-08520893ed4f.jpg)
![left](https://user-images.githubusercontent.com/84385239/165281874-971fe057-b1dc-42d7-8984-73a1f3e14180.jpg)

to normal map textures like this:
![normalmap](https://user-images.githubusercontent.com/84385239/165281891-a7989a2c-9d74-4707-96c6-95bb3524c92c.png)


You can use just *hm2nm(img)* function to create a normal map from a single RGB image, or just *ps2nm(imgU, imgD, imgR, imgL)* to create more realistic normal maps, or combine them together to create normal maps with much more detailed edges.

Also **overlay blend mode** is implemented to combine two normal maps.
