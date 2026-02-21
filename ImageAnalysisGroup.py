#__author__ = João Martins (ist1106976)
#__author__ = Pedro Ramos (ist1106010)


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import os
from ImageAnalysis import ImageAnalysis 

class ImageAnalysisGroup:
    def __init__(self, image_analysis_list):
        self.image_analysis_list = image_analysis_list

    def convolve_with_itself_mean(self, data_type='filtered', min_threshold=None):
        print("begin convolve_with_itself_mean")
        convolved_matrices = []

        for analysis in self.image_analysis_list:
            interval_filtered_data = analysis.filter_pixels_within_interval(data_type)
            convolved_data = fftconvolve(interval_filtered_data, interval_filtered_data, mode='same')

            # Normalize the convolved data to be between 0 and 1
            convolved_data_min = np.min(convolved_data)
            convolved_data_max = np.max(convolved_data)
            convolved_data_normalized = (convolved_data - convolved_data_min) / (convolved_data_max - convolved_data_min)

            if min_threshold is not None:
                convolved_data_normalized[convolved_data_normalized < min_threshold] = 0  # Apply minimum value threshold

            convolved_matrices.append(convolved_data_normalized)

        # Compute the mean of the convolved matrices
        mean_convolved_data = np.mean(convolved_matrices, axis=0)

        # Normalize the mean convolved data to be between 0 and 1
        mean_convolved_data_min = np.min(mean_convolved_data)
        mean_convolved_data_max = np.max(mean_convolved_data)
        mean_convolved_data_normalized = (mean_convolved_data - mean_convolved_data_min) / (mean_convolved_data_max - mean_convolved_data_min)

        print("end convolve_with_itself_mean")
        return mean_convolved_data_normalized

    def plot_heatmap_mean(self, data_type='filtered', title='Heatmap of Mean Convolved Data', log_scale=False, min_threshold=None):
        print("begin plot_heatmap_mean")
        mean_convolved_data = self.convolve_with_itself_mean(data_type, min_threshold)

        if log_scale:
            mean_convolved_data = np.log(mean_convolved_data + 1)  # Apply logarithmic scale
        
        plt.figure(figsize=(10, 10))
        plt.imshow(mean_convolved_data, interpolation='nearest', cmap='coolwarm')
        plt.colorbar()
        plt.title(title)
        plt.axis('off')  # Hide the axis
        path_fig = os.getcwd()
        plt.savefig(os.path.join(path_fig, f'{title.replace(" ", "_")}.png'))
        print("end plot_heatmap_mean")
        plt.show()

    def plot_fourier_transform_mean(self, plot_dft_squared=True, data_type='filtered', title='Fourier Transform of Mean Convolved Data', min_threshold=None):
        print("begin plot_fourier_transform_mean")
        mean_convolved_data = self.convolve_with_itself_mean(data_type, min_threshold)
        fourier_transform = self.image_analysis_list[0].compute_fourier_transform(mean_convolved_data)
        
        if plot_dft_squared:
            data_to_plot = self.image_analysis_list[0].compute_dft_squared(fourier_transform)
            data_to_plot = np.log(data_to_plot + 1)
            title += ' DFT_squared'
        else:
            data_to_plot = np.log(np.abs(fourier_transform) + 1)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(data_to_plot, cmap='gray')
        plt.colorbar()
        plt.title(title)
        plt.axis('off')  # Hide the axis
        path_fig = os.getcwd()
        plt.savefig(os.path.join(path_fig, f'{title.replace(" ", "_")}.png'))
        print("end plot_fourier_transform_mean")
        plt.show()

    def plot_total_hist(self, data_type='filtered', title='Total histogram of filtered data', scale=False):
        print("begin plot_total_hist")
        total_data = self.image_analysis_list[0].filtered_data

        for i in range(1, len(self.image_analysis_list)):
            total_data += self.image_analysis_list[i].filtered_data

        treshold = 1
        data2 = total_data[total_data>treshold]

        #maximo do data 2
        data_max = int(np.max(data2))
        
        # Calculate the histogram data
        counts, bins = np.histogram(data2.flatten(), bins=data_max)

        fig, axs = plt.subplots(1, 1, figsize=(12, 8))
        # Plot a line according to the histogram
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        axs.plot(bin_centers, counts, color='red', linestyle='-', linewidth=1, label='Data Line')

        axs.set_title(title)
        axs.set_xlabel("Intensity")
        axs.set_ylabel("Counts")
        if not scale:
            axs.set_yscale('log')
        axs.set_xlim(0, data_max )  # Adjust x-axis limit to show just the tip of the histogram
        axs.grid(axis='y', alpha=0.75)
        axs.legend()

         # Adjust layout to prevent overlap
        plt.tight_layout()
        path_fig = os.getcwd()

        plt.savefig(os.path.join(path_fig, f'{title.replace(" ", "_")}.png'))