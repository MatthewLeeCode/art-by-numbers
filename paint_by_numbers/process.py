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
    masks: list=[]
    contours: list=[]
    heirarchies: list=[]
    shells: list=[]
    holes: list=[]
    cluster_labels: dict=None
    cluster_image: np.ndarray=None
    simplified_image: np.ndarray=None
    pbn_image: np.ndarray=None
    background_color: tuple=(255, 255, 255)
    contour_color: tuple=(0, 0, 0)
    label_color: tuple=(0, 0, 0)
    contour_thickness: int=1
    label_thickness: int=1
    label_font: int=cv2.FONT_HERSHEY_SIMPLEX
    label_positions: list=[]
    currently_processing_label:int = None
    
    def __init__(self, 
        image: np.ndarray, 
        num_colors:int=10, 
        height:int=None, 
        width:int=2000, 
        min_area:int=20,
        custom_colors:list=[],
        strip_removal_count=2) -> None:
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
        # Resize to match width but keep aspect ratio
        if height is None:
            height = int(image.shape[0] * (width / image.shape[1]))
        elif width is None:
            width = int(image.shape[1] * (height / image.shape[0]))
        elif height is None and width is None:
            height = image.shape[0]
            width = image.shape[1]
        
        assert len(custom_colors) <= num_colors, "The number of custom colors must be less than or equal to the number of colors."
        self.num_colors = num_colors
        self.height = height
        self.width = width
        self.min_area = min_area
        self.custom_colors = custom_colors  
        self.strip_removal_count = strip_removal_count
        
        # Resize image
        self.image = image
        self.image = cv2.resize(image, (self.width, self.height))
    
    def cluster_image(self) -> dict:
        """ Cluster the image into the num_colors
        
        Assigns the simplified_image, cluster_labels, and cluster_image
        
        Returns:
            The cluster labels
        """
        # Cluster the image
        clustering_spinner = Halo(text='Clustering image...', spinner='dots')
        clustering_spinner.start()
        
        self.simplified_image = clustering.blur_image(image=self.image)
        #self.simplified_image = clustering.morphology(self.simplified_image)
        
        self.cluster_labels, self.cluster_image = clustering.kmeans_cluster(image=self.simplified_image, k=self.num_colors)
        
        # Strip removal. Best to do it a few times
        distance_matrix = clustering.create_distance_matrix(color_labels=self.cluster_labels)
        for _ in range(self.strip_removal_count):
            self.cluster_image = clustering.remove_strips(
                color_labels=self.cluster_labels, 
                image=self.cluster_image,
                distance_matrix=distance_matrix
            )
            
        clustering_spinner.succeed('Image clustered.')
        return self.cluster_labels
    
    def find_mask(self, color:tuple) -> np.ndarray:
        """ Find the mask for a color
        
        Args:
            color (tuple): The RGB color to find the mask for
            
        Returns:
            np.ndarray: The mask
        """
        with Halo(text=f'Finding mask for label {self.currently_processing_label}...', spinner='dots'):
            mask = clustering.create_mask(clustered_image=self.cluster_image, rgb=color)
        return mask
    
    def find_contours(self, mask:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ Find the contours for a mask
        
        Args:
            mask (np.ndarray): The mask to find contours for
            
        Returns:
            tuple: The contours and hierarchy
        """
        with Halo(text=f'Finding contours for label {self.currently_processing_label}...', spinner='dots'):
            contours, hierarchy = contouring.get_mask_contours(mask=mask)
        return contours, hierarchy
    
    def find_shells_and_holes(self, contours:np.ndarray, hierarchy:np.ndarray) -> tuple[list, list]:
        """ Find the shells and holes for a set of contours
        
        Args:
            contours (np.ndarray): The contours to find shells and holes for
            hierarchy (np.ndarray): The hierarchy of the contours
            
        Returns:
            tuple: The shells and holes
        """
        with Halo(text=f'Finding contour shells and holes for label {self.currently_processing_label}...', spinner='dots'):
            shells, holes = contouring.find_shell_holes(contours=contours, hierarchy=hierarchy)
        return shells, holes
    
    def prune_contours(self, shells:list[np.ndarray], holes:list[list[np.ndarray]]) -> tuple[list, list]:
        """ Prune the contours
        
        Args:
            shells (list[np.ndarray]): The shells to prune
            holes (list[list[np.ndarray]]): The holes to prune
            
        Returns:
            tuple: The pruned shells and holes
        """
        with Halo(text=f'Pruning contours for label {self.currently_processing_label}...', spinner='dots'):
            shells, holes = contouring.remove_small_contours(shells=shells, holes=holes, min_area=self.min_area)
        return shells, holes
    
    def find_label_positions(self, shells:list[np.ndarray], shell_holes:list[list[np.ndarray]]) -> tuple[list, list]:
        """ Get the label positions for a set of shells and holes
        
        Args:
            shells (list[np.ndarray]): The shells to find label positions for
            shell_holes (list[list[np.ndarray]]): The holes to find label positions for

        Returns:
            tuple: A list of label positions and a list of scales
        """
        with Halo(text=f'Finding label positions for label {self.currently_processing_label}...', spinner='dots'):
            label_positions = []
            label_scales = []
            for shell, holes in zip(shells, shell_holes):
                label_position, distance = labelling.find_visual_center(shell=shell, holes=holes)
                if label_position is not None:
                    label_positions.append(label_position)
                    # We need to center the text on the position
                    text_size = cv2.getTextSize(str(self.currently_processing_label), self.label_font, 1, self.label_thickness)
                    # If the text is wider than the distance, we need to scale it down
                    scale = 1
                    if text_size[0][0] > distance:
                        scale = distance / text_size[0][0]
                    
                    label_scales.append(scale)

        return label_positions, label_scales
    
    def draw_labels(self, image: np.ndarray, label:str, label_positions:list, label_scales:list) -> np.ndarray:
        """ Draw the labels on the image
        
        Args:
            image (np.ndarray): The image to draw on
            label (str): The label to draw
            label_positions (list): The positions of the labels
            label_scales (list): The scales of the labels
            
        Returns:
            np.ndarray: The image with labels drawn on it
        """
        with Halo(text=f'Drawing contours and labels for label {self.currently_processing_label}...', spinner='dots'):
            # Draw labels expects a list of labels + positions with both lists being the same size.
            # So we simply just expand the label to be a list of the same size as the label_positions
            labels = [label] * len(label_positions)
            
            # Draw labels
            image = drawing.draw_labels(
                frame=image,
                labels=labels,
                positions=label_positions,
                color=self.label_color,
                font=self.label_font,
                scales=label_scales,
                thickness=self.label_thickness
            )
        return image
    
    
    def draw_color_regions(self, image: np.ndarray, color:tuple, shells:list[np.ndarray], shell_holes:list[list[np.ndarray]]) -> np.ndarray:
        """ Draw the color regions on the image
        
        Args:
            image: The image to draw on
            color: The color to draw
            shells: The shells to draw
            shell_holes: The holes to draw
            
        Returns:
            The image with the color regions drawn on it
        """
        with Halo(text=f'Drawing color regions for label {self.currently_processing_label}...', spinner='dots'):
            image = drawing.draw_color_regions(
                image=image,
                color=color,
                shells=shells,
                shell_holes=shell_holes
            )
        return image
        
    def draw_contours(self, image:np.ndarray, contours:list[np.ndarray]) -> np.ndarray:
        """ Draw the contours on the image
        
        Args:
            image: The image to draw the contours
            contours: The contours to draw
            
        Returns:
            The image with the contours drawn on it
        """
        with Halo(text=f'Drawing contours for label {self.currently_processing_label}...', spinner='dots'):
            image = cv2.drawContours(
                image=image,
                contours=contours,
                contourIdx=-1,
                color=self.contour_color,
                thickness=self.contour_thickness
            )
        return image    
    
    def process(self) -> np.ndarray:
        """ 
        Process the image.
        
        Steps:
        1. Cluster the image into k colors (Includes removing noise)
        2. Create a mask for each color in the image
        3. Find the contours for each mask
        4. Prune the contours to remove small contours
        5. Find a label position for each contour
        6. Draw the contours and labels on the image
        
        Assigns values for:
        - simplified_image (np.ndarray): The simplified image after noise reduction 
        - cluster_labels (dict): The labels for each cluster
        - cluster_image (np.ndarray): The clustered image 
        - pbn_image (np.ndarray): The processed image with contours and labels drawn on it
        - label_positions (list): The positions of the labels

        Returns:
            np.ndarray: The processed image.
        """
        self.mask = []
        self.contours = []
        self.heirarchies = []
        self.shells = []
        self.holes = []
        
        # 1. Cluster the image into k colors (Includes removing noise)
        self.cluster_image()
        
        # Create an empty frame
        self.pbn_image = drawing.get_frame(height=self.height, width=self.width)
        self.pbn_color = drawing.get_frame(height=self.height, width=self.width) # Color is the color regions
        
        # Order the keys so that the labels are drawn in the correct order
        label_keys = self.cluster_labels.keys()
        label_keys = sorted(label_keys, key=lambda x: int(x))
        
        # Loop through each color
        for key in label_keys:
            self.currently_processing_label = key
            success_spinner = Halo(text=f'Label {key} processing...', spinner='dots')
            try:
                # 2. Find mask for each color
                mask = self.find_mask(color=self.cluster_labels[key])
                self.masks.append(mask)
                
                # 3. Find contours
                contours, hierarchy = self.find_contours(mask=mask)
                self.contours.append(contours)
                self.heirarchies.append(hierarchy)
                
                shells, holes = self.find_shells_and_holes(contours=contours, hierarchy=hierarchy)
                
                # 4. Prune contours
                shells, holes = self.prune_contours(shells=shells, holes=holes)
                self.shells.append(shells)
                self.holes.append(holes)
                
                # 5. Find label position
                label_positions, label_scales = self.find_label_positions(shells=shells, shell_holes=holes)
                
                # 6. Draw contours and labels
                self.pbn_color = self.draw_color_regions(self.pbn_color, self.cluster_labels[key], shells, holes)
                self.pbn_image = self.draw_labels(self.pbn_image, key, label_positions, label_scales)
                self.pbn_image = self.draw_contours(self.pbn_image, contours)
            
            except Exception as e:
                # Add success spinner message for label
                success_spinner.fail(f'Label {self.currently_processing_label} failed to process.')
                raise e
            
            # Successful completion of this mask
            success_spinner.succeed(f'Label {self.currently_processing_label} processed.')
            
        # Return the final paint-by-numbers image
        return self.pbn_image
                
            
            