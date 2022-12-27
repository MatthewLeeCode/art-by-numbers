"""
Class for processing the paint-by-numbers image.

Performs the following steps:
- Clustering the image into k colors
- Masking the image for each color
- Find the contours for each mask
- Find a label position for each image
- Draw the contours and labels on the image
"""
import numpy as np
import clustering
import contouring
import labelling
import drawing
import cv2
from halo import Halo


class PaintByNumbers:
    cluster_labels: dict=None
    cluster_image: np.ndarray=None
    simplified_image: np.ndarray=None
    pbn_image: np.ndarray=None
    background_color: tuple=(255, 255, 255)
    contour_color: tuple=(0, 0, 0)
    label_color: tuple=(0, 0, 0)
    contour_thickness: int=1
    label_thickness: int=1
    label_scale: float=1
    label_font: int=cv2.FONT_HERSHEY_SIMPLEX
    label_positions: list=[]
    
    def __init__(self, 
        image: np.ndarray, 
        num_colors:int=10, 
        height:int=None, 
        width:int=None, 
        min_area:int=50,
        custom_colors:list=[]) -> None:
        """ 
        Initialize the class.

        Args:
            image (np.ndarray): The image to process.
            num_colors (int, optional): The number of colors to cluster the image into. Defaults to 10.
            height (int, optional): The height of the output image. Defaults to None.
            width (int, optional): The width of the output image. Defaults to None.
            min_area (int, optional): The minimum area for a contour to be considered. Defaults to 50.
            custom_colors (list, optional): A list of custom colors to use for the labels. Defaults to [].
        """
        if height is None:
            height = image.shape[0]
        
        if width is None:
            width = image.shape[1]
        
        assert len(custom_colors) <= num_colors, "The number of custom colors must be less than or equal to the number of colors."
        self.num_colors = num_colors
        self.height = height
        self.width = width
        self.min_area = min_area
        self.custom_colors = custom_colors  
        
        # Resize image
        self.image = cv2.resize(image, (self.width, self.height))
        
    def process(self) -> np.ndarray:
        """ 
        Process the image. Creates the paint-by-numbers image and stores
        the result in the pbn_image attribute. Additionally, it also sets
        the cluster_labels and cluster_image attributes.

        Returns:
            np.ndarray: The processed image.
        """
        # Cluster the image
        clustering_spinner = Halo(text='Clustering image...', spinner='dots')
        clustering_spinner.start()
        
        self.simplified_image = clustering.remove_noise(image=self.image)
        self.cluster_labels, self.cluster_image = clustering.kmeans_cluster(image=self.simplified_image, k=self.num_colors)
        
        clustering_spinner.succeed('Image clustered.')
        
        # Loop through each mask and draw the contours and labels to frame
        # Create an empty frame
        self.pbn_image = drawing.get_frame(height=self.height, width=self.width)
        
        for key in self.cluster_labels.keys():
            
            # Find mask
            mask_spinner = Halo(text=f'Processing mask for label {key}...', spinner='dots')
            mask_spinner.start()
            
            mask = clustering.create_mask(clustered_image=self.cluster_image, rgb=self.cluster_labels[key])
            
            mask_spinner.succeed(f'Mask for label {key} processed.')
            
            # Find contours
            contour_spinner = Halo(text=f'Finding contours for label {key}...', spinner='dots')
            contour_spinner.start()
            
            contours, hierarchy = contouring.get_mask_contours(mask=mask, min_area=self.min_area)
            
            contour_spinner.succeed(f'Contours for label {key} found.')
            
            # Find shells (outer contours) and holes (inner contours)
            polygon_spinner = Halo(text=f'Finding shells and holes for label {key}...', spinner='dots')
            polygon_spinner.start()
            
            polygons = contouring.find_shell_holes(contours=contours, hierarchy=hierarchy)
            
            polygon_spinner.succeed(f'Shells and holes for label {key} found.')

            # Find label position
            label_spinner = Halo(text=f'Finding label positions for label {key}...', spinner='dots')
            label_spinner.start()
            
            self.label_positions = []
            for polygon in polygons:
                # First element in polygon is the shell
                # Second element in polygon is the holes
                label_position = labelling.find_visual_center(shell=polygon[0], holes=polygon[1])
                self.label_positions.append(label_position)
            
            label_spinner.succeed(f'Label positions for label {key} found.')
            
            # Draw contours and labels
            draw_spinner = Halo(text=f'Drawing contours and labels for label {key}...', spinner='dots')
            draw_spinner.start()
            
            drawing.draw_contours(
                image=self.pbn_image, 
                contours=contours, 
                color=self.contour_color, 
                thickness=self.contour_thickness
            )
            
            # Draw labels expects a list of labels + positions with both lists being the same size.
            # So we simply just expand the label to be a list of the same size as the label_positions
            labels = [self.cluster_labels[key]] * len(self.label_positions)
            
            # Draw labels
            drawing.draw_labels(
                image=self.pbn_image,
                labels=labels,
                positions=self.label_positions,
                color=self.label_color,
                font=self.label_font,
                scale=self.label_scale,
                thickness=self.label_thickness
            )
            
            draw_spinner.succeed(f'Contours and labels for label {key} drawn.')
            
            # Clear spinners
            mask_spinner.stop()
            contour_spinner.stop()
            polygon_spinner.stop()
            label_spinner.stop()
            draw_spinner.stop()
            
            # Add success spinner message for label
            success_spinner = Halo(text=f'Label {key} processed.', spinner='dots')
            success_spinner.succeed()
            
        # Return the final paint-by-numbers image
        return self.pbn_image
                
            
            