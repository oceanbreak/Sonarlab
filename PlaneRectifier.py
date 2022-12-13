"""
This module provides tools for rectification of videhosnapshots of planar surfaces
based on fundamental matrix.
Workflow is such: to consequent images are chosen, matching feauters are found,
then essential matrix os found and decomposed into R and T matricies.
Based on R and T triangulations for found points is made, so points in 3D are found.
Then a plane that fits all the points is found. Then original image is rotated based on direction 
of the normal of the found plane, so then image becomes planar and one may accurately calculate linear sizes.
"""

