# Object-Detection

## Workflow abstraction
1. Find the horizontal lines (find the first line in the 5 parallel lines) and distance ($d$) between lines.
    * Hough transform is used.
2. Find the template
    1. Make a better template - resize template corresponding to $d$ by comparing template 1's height and $d$.
    2. Template matching
        * General workflow
            1. Detected the horizontal lines using hough transform.
            2. Pre-process image by converting it into greyscale.
            3. Make a probability map by using cross correlatin between the image and the 3 templates. 
            4. Performed non-max suppression.
            5. Got the probabilities and took the top 20% high probability values giving the most probable regions where the templates can be found.
            6. Get the center of each instance.
            7. Make a bounding box (with the size of template) around the center.
            8. Labelled the bounding boxes with the corresponding note with the help of the horizontal lines detected in the first step.
            9. Final results are plot.

## Results
![image](https://user-images.githubusercontent.com/60294261/227747052-d77b08e0-0756-4de2-a9cb-79ff50526a67.png)

![image](https://user-images.githubusercontent.com/60294261/227747118-5822772d-3819-4723-b5a0-54f81073160d.png)

![image](https://user-images.githubusercontent.com/60294261/227747128-8a6993ce-8104-4bfd-b6b5-dc5e4055979a.png)


## Improvements
The image can be furthur pre processed to remove the horizontal lines for better object detection using image segmentation.


