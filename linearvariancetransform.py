__doc__ ="""\
VarianceTransform
=================
**VarianceTransform** 
This module allows you to calculate the variance of an image, using a determined window size. It also has
the option to find the optimal window size from a predetermied range to obtain the maximum variance of an image.
============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES           YES
============ ============ ===============
"""

import numpy as np
import scipy.ndimage

import cellprofiler_core.setting
import cellprofiler_core.module
from cellprofiler_core.image import Image
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import ImageName, Integer


class LinearVarianceTransform(cellprofiler_core.module.ImageProcessing):
    module_name = "LinearVarianceTransform"

    variable_revision_number = 1
    def create_settings(self):
        self.image_name = ImageSubscriber(
            "Select the input image",
            "None",
            doc="""Select the image to be smoothed.""",
        )

        self.output_image_name = ImageName(
            "Name the output image",
            "FilteredImage",
            doc="""Enter a name for the resulting image.""",
        )

        self.calculate_maximal = Binary(
                "Calculate optimal window size to maximize image variance?",
                False,
                doc="""\
Select "*Yes*" to provide a range that will be used to obtain the window size that will generate
the maximum variance in the input image.
Select "*No*" to give the window size used to obtain the image variance.""",
        )

        self.window_size = Integer(
            "Window size",
            5, 
            minval=1,
            doc="""Enter the size of the window used to calculate the variance.""",
        )

        self.window_min = Integer(
            "Window min",
            5, 
            minval=1,
            doc="""Enter the minimum size of the window used to calculate the variance.""",
        )

        self.window_max = Integer(
            "Window max",
            50, 
            minval=1,
            doc="""Enter the maximum size of the window used to calculate the variance.""",
        )


    def settings(self):
        return [
            self.image_name,
            self.output_image_name,
            self.calculate_maximal,
            self.window_size,
            self.window_min,
            self.window_max,

        ]


    def visible_settings(self):
        __settings__ = [self.image_name, self.output_image_name,]
        __settings__ += [self.calculate_maximal,]
        if not self.calculate_maximal.value:
            __settings__ += [self.window_size,]
        else:
            __settings__ += [self.window_min, self.window_max,]
        return __settings__

    def run(self, workspace):
        
        image = workspace.image_set.get_image(self.image_name.value, must_be_grayscale=True)
        
        image_pixels = image.pixel_data

        shape = np.array(image_pixels.shape).astype(int)

        num_cols, num_rows = shape[-2], shape[-1]

        window_range = range(self.window_min.value,self.window_max.value,1)
        
        size = self.window_size.value

        if self.calculate_maximal.value:
            max_variance = -1
            size = -1
            for window in window_range:
                window_xtra = int((window - 1) / 2)
                results_list = []
                for col_num in range(num_cols):
                    row_results = []
                    for row_num in range(window_xtra, num_rows):
                        row_values = image_pixels[row_num - window_xtra:row_num + window_xtra, col_num]
                        mean = np.mean(row_values)
                        result = np.mean((row_values - mean) ** 2)

                        row_results.append(result)
                    results_list.append(row_results)

                variance = max(max(inner_list) for inner_list in results_list)
                if variance > max_variance:
                    max_variance = variance
                    size = window

        output_pixels = np.zeros_like(image_pixels)
        size_xtra = int((size - 1) / 2)
        for col_num in range(num_cols):
            for row_num in range(0, num_rows):
                row_values = image_pixels[row_num-size_xtra:row_num+size_xtra, col_num]
                mean = np.mean(row_values)
                result = np.mean((row_values - mean) ** 2)
                output_pixels[row_num, col_num] = result

        new_image = Image(output_pixels, parent_image=image, dimensions=image.dimensions)
            
        workspace.image_set.add(self.output_image_name.value, new_image)
        
        if self.show_window:
            workspace.display_data.pixel_data  = image_pixels

            workspace.display_data.output_pixels= output_pixels

            workspace.display_data.dimensions = image.dimensions

    def display(self, workspace, figure):
        layout = (2, 1)
        figure.set_subplots(
            dimensions=workspace.display_data.dimensions, subplots=layout
        )

        figure.subplot_imshow(
            colormap="gray",
            image = workspace.display_data.pixel_data,
            title = self.image_name.value,
            x=0,
            y=0,
        )

        figure.subplot_imshow(
            colormap="gray",
            image = workspace.display_data.output_pixels,
            title = self.output_image_name.value,
            x=1,
            y=0,
        )
