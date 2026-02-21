#__author__ = João Martins (ist1106976)
#__author__ = Pedro Ramos (ist1106010)


import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import os
from scipy.signal import fftconvolve



class ImageAnalysis:

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = imread(file_path)
        self.data_max = int(np.max(self.data))
        self.filtered_data = ImageAnalysis.analyze_neighbors(self) #single photons
        self.summed_data = ImageAnalysis.analyze_neighbors_2(self)


    def analyze_neighbors(self):
        shape = self.data.shape
        filtered_data = np.zeros(shape)

        for i in range(shape[0]):
            for j in range(shape[1]):
                # Initialize a list to store the values of the 8 neighbors
                neighbors = []

                # Loop through the 3x3 grid centered at (i, j)
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        # Skip the center pixel
                        if di == 0 and dj == 0:
                            continue

                        # Calculate the neighbor's coordinates
                        ni, nj = i + di, j + dj

                        # Check if the neighbor is within the bounds of the array
                        if 0 <= ni < shape[0] and 0 <= nj < shape[1]:
                            neighbors.append(self.data[ni, nj])

                # Perform your analysis on the neighbor
                if neighbors: 
                    neighbors_sum = np.sum(neighbors)
                    if self.data[i, j] > 1.98 * neighbors_sum  and  self.data[i, j] + neighbors_sum > 500 :
                        filtered_data[i, j] = self.data[i, j] + neighbors_sum

        return filtered_data


    def analyze_neighbors_2(self):
        shape = self.data.shape
        filtered_data = np.zeros(shape)

        for i in range(shape[0]):
            for j in range(shape[1]):
                # Initialize a list to store the values of the 8 neighbors
                neighbors = []

                # Loop through the 3x3 grid centered at (i, j)
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        # Skip the center pixel
                        if di == 0 and dj == 0:
                            continue

                        # Calculate the neighbor's coordinates
                        ni, nj = i + di, j + dj

                        # Check if the neighbor is within the bounds of the array
                        if 0 <= ni < shape[0] and 0 <= nj < shape[1]:
                            neighbors.append(self.data[ni, nj])

                # Perform your analysis on the neighbor
                if neighbors: 
                    neighbors_sum = np.sum(neighbors)
                    filtered_data[i, j] = self.data[i, j] + neighbors_sum

        return filtered_data


    def hist_original_vs_single(self, scale = True, title = 'Histogram of Pixel Intensities'):
        # Create the subplot (1 row, 1 column)

        #data_max = int(np.max(self.data))

        fig, axs = plt.subplots(1, 1, figsize=(12, 8))

        # Calculate the histogram data
        counts, bins = np.histogram(self.data.flatten(), bins=self.data_max)

        filtered_counts, filtered_bins = np.histogram(self.filtered_data.flatten(), bins=self.data_max)

        # Apply the threshold to consider only counts below 1000
        threshold = 500
        counts = counts[threshold:]
        filtered_counts = filtered_counts[threshold:]

        # Plot a line according to the histogram
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        axs.plot(bin_centers, counts, color='red', linestyle='-', linewidth=1, label='Original Data Line')

        filtered_bin_centers = 0.5 * (filtered_bins[:-1] + filtered_bins[1:])
        axs.plot(filtered_bin_centers, filtered_counts, color='blue', linestyle='-', linewidth=1, label='Filtered Data Line')

        axs.set_title(title)
        axs.set_xlabel("Intensity")
        axs.set_ylabel("Counts")
        if not scale:
            axs.set_yscale('log')
        axs.set_xlim(0, self.data_max)  # Adjust x-axis limit to show just the tip of the histogram
        axs.grid(axis='y', alpha=0.75)
        axs.legend()

        # Adjust layout to prevent overlap
        plt.tight_layout()
        path_fig = os.getcwd()
        plt.savefig(os.path.join(path_fig, f'{title.replace(" ", "_")}_hist_filtered_vs_single.png'))


    def hist_single(self, title='Histogram of Pixel Intensities',  data_type='filtered',scale=True):
        # Create the subplot (1 row, 1 column)
        fig, axs = plt.subplots(1, 1, figsize=(12, 8))
        
        if data_type == 'filtered':
            data = self.filtered_data
        elif data_type == 'summed':
            data = self.summed_data
        elif data_type == 'original':
            data = self.data
        else:
            raise ValueError("Invalid data_type. Expected 'filtered', 'summed', or 'original'.")

        treshold = 500
        data2 = data[data>treshold]
        
        # Calculate the histogram data
        counts, bins = np.histogram(data2.flatten(), bins=self.data_max)

        # Plot a line according to the histogram
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        axs.plot(bin_centers, counts, color='red', linestyle='-', linewidth=1, label='Data Line')

        axs.set_title(title)
        axs.set_xlabel("Intensity")
        axs.set_ylabel("Counts")
        if not scale:
            axs.set_yscale('log')
        axs.set_xlim(0, self.data_max)  # Adjust x-axis limit to show just the tip of the histogram
        axs.grid(axis='y', alpha=0.75)
        axs.legend()

        # Adjust layout to prevent overlap
        plt.tight_layout()
        path_fig = os.getcwd()
        
        if data_type == 'filtered':
            if scale:
                plt.savefig(os.path.join(path_fig, f'{title.replace(" ", "_")}_hist_single_log_filtered.png'))
            else:
                plt.savefig(os.path.join(path_fig, f'{title.replace(" ", "_")}_hist_single_filtered.png'))
        elif data_type == 'summed':
            if scale:
                plt.savefig(os.path.join(path_fig, f'{title.replace(" ", "_")}_hist_single_log_summed.png'))
            else:
                plt.savefig(os.path.join(path_fig, f'{title.replace(" ", "_")}_hist_single_summed.png'))
        elif data_type == 'original':
            if scale:
                plt.savefig(os.path.join(path_fig, f'{title.replace(" ", "_")}_hist_single_log_original.png'))
            else:
                plt.savefig(os.path.join(path_fig, f'{title.replace(" ", "_")}_hist_single_original.png'))


    def Histogram_2D(self, title = 'Histogram of Pixel Intensities', data_type='filtered'):
        if data_type == 'filtered':
            data = self.filtered_data
        elif data_type == 'summed':
            data = self.summed_data
        elif data_type == 'original':
            data = self.data
        else:
            raise ValueError("Invalid data_type. Expected 'filtered', 'summed', or 'original'.")

        plt.figure(figsize=(10, 10))
        plt.imshow(data, interpolation='nearest')
        plt.title("Black and White Image")
        plt.axis('off')  # Hide the axis

        path_fig = os.getcwd()
        
        if data_type == 'filtered':
            plt.savefig(os.path.join(path_fig, f'{title.replace(" ", "_")}_hist2D_filtered.png'))
        elif data_type == 'summed':
            plt.savefig(os.path.join(path_fig, f'{title.replace(" ", "_")}hist2D_summed.png'))
        elif data_type == 'original':
            plt.savefig(os.path.join(path_fig, f'{title.replace(" ", "_")}_hist2D_original.png'))


    def difference_Hist_2D(self, title = 'Difference between Original and Filtered Data', s='filtered', s2='original'):
        if s == 'filtered':
            data = self.filtered_data
        elif s == 'summed':
            data = self.summed_data
        elif s == 'original':
            data = self.data
        
        if s2 == 'filtered':
            data2 = self.filtered_data
        elif s2 == 'summed':
            data2 = self.summed_data
        elif s2 == 'original':
            data2 = self.data
            
        diff = data - data2

        plt.figure(figsize=(10, 10))
        plt.imshow(diff, interpolation='nearest')
        plt.title(title)
        plt.axis('off')

        path_fig = os.getcwd()
        plt.savefig(os.path.join(path_fig + f'_difference_{s}_{s2}.png'))
        plt.show()

    
    def Kalphas(self, data_type='filtered', n1=1250, n2=1750):
        if data_type == 'filtered':
            data = self.filtered_data
        elif data_type == 'summed':
            data = self.summed_data
        elif data_type == 'original':
            data = self.data

        if n1 < 0 or n2 > len(data):
            raise ValueError("n1 and n2 must be within the bounds of the data array")

        # Restrict the data to the specified range
        # Flatten the restricted data to 1D
        flattened_data = data.flatten()

        # Calculate the histogram data
        counts, bins = np.histogram(flattened_data, bins=self.data_max)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
        # Restrict the bins and counts to the specified range
        valid_indices = (bin_centers >= n1) & (bin_centers <= n2)
        counts = counts[valid_indices]
        bin_centers = bin_centers[valid_indices]

        # Calculate the weighted mean of the restricted data
        weighted_mean_intensity = np.average(bin_centers, weights=counts)
        std_dev = np.sqrt(np.average((bin_centers - weighted_mean_intensity)**2, weights=counts))

        return weighted_mean_intensity, std_dev


    def single_photon_intensity_distance(self):
        shape = self.data.shape
        layer_means = np.zeros((shape[0], shape[1], 5))

        for i in range(shape[0]):
            for j in range(shape[1]):
                if self.filtered_data[i, j] == 0:
                    continue

                for layer in range(1, 6):
                    neighbors = []
                    for di in range(-layer, layer + 1):
                        for dj in range(-layer, layer + 1):
                            if di == 0 and dj == 0:
                                continue

                            ni, nj = i + di, j + dj
                            if 0 <= ni < shape[0] and 0 <= nj < shape[1]:
                                neighbors.append(self.data[ni, nj])

                    if neighbors:
                        layer_means[i, j, layer - 1] = np.mean(neighbors) / self.data[i, j]

        # Calculate the mean fraction for each layer
        mean_fractions = np.mean(layer_means, axis=(0, 1))

        # Plot the mean fraction as a function of the layer
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 6), mean_fractions, marker='o', linestyle='-', color='b')
        plt.title('Mean Fraction of Intensity as a Function of the Layer')
        plt.xlabel('Layer')
        plt.ylabel('Mean Fraction of Intensity')
        plt.grid(True)
        path_fig = os.getcwd()
        plt.savefig(os.path.join(path_fig, 'mean_fraction_vs_layer.png'))
        plt.show()


    def filter_pixels_within_interval(self, data_type='filtered'):
        print("begin filter_pixels_within_interval")
        # Get the mean and standard deviation from the Kalphas function
        mean, std = self.Kalphas(data_type)

        if data_type == 'filtered':
            data = self.filtered_data
        elif data_type == 'summed':
            data = self.summed_data
        elif data_type == 'original':
            data = self.data
        else:
            raise ValueError("Invalid data_type. Expected 'filtered', 'summed', or 'original'.")

        # Define the interval
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std

        # Create a new array with the same shape as the data
        interval_filtered_data = np.zeros(data.shape)

        # Set pixels within the interval to 1
        interval_filtered_data[(data >= lower_bound) & (data <= upper_bound)] = 1
        
        print("end filter_pixels_within_interval")
        return interval_filtered_data


    def convolve_with_itself(self, data_type='filtered'):
        print("begin convolve_with_itself")
        interval_filtered_data = self.filter_pixels_within_interval(data_type)
        convolved_data = fftconvolve(interval_filtered_data, interval_filtered_data, mode='same')

         # Normalize the convolved data to be between 0 and 1
        convolved_data_min = np.min(convolved_data)
        convolved_data_max = np.max(convolved_data)
        convolved_data_normalized = (convolved_data - convolved_data_min) / (convolved_data_max - convolved_data_min)
    
        print("end convolve_with_itself")
        return convolved_data_normalized


    def convolve_with_dft_squared(self, data_type='filtered'):
        print("begin convolve_with_dft_squared")
        interval_filtered_data = self.filter_pixels_within_interval(data_type)
        convolved_data = fftconvolve(interval_filtered_data, interval_filtered_data, mode='same')

        # Normalize the convolved data to be between 0 and 1
        convolved_data_min = np.min(convolved_data)
        convolved_data_max = np.max(convolved_data)
        convolved_data_normalized = (convolved_data - convolved_data_min) / (convolved_data_max - convolved_data_min)

        # Compute the Fourier Transform
        fourier_transform = self.compute_fourier_transform(convolved_data_normalized)

        # Compute |DFT|^2
        dft_squared = self.compute_dft_squared(fourier_transform)

        # Perform the inverse Fourier Transform
        new_convolved_data = self.inverse_fourier_transform(dft_squared)

        # Normalize the new convolved data to be between 0 and 1
        new_convolved_data_min = np.min(new_convolved_data)
        new_convolved_data_max = np.max(new_convolved_data)
        new_convolved_data_normalized = (new_convolved_data - new_convolved_data_min) / (new_convolved_data_max - new_convolved_data_min)

        print("end convolve_with_dft_squared")
        return new_convolved_data_normalized


    def plot_heatmap(self, data_type='filtered', title='Heatmap of Convolved Data', use_dft=False, log_scale=False):
        print("begin plot_heatmap")
        if use_dft:
            convolved_data = self.convolve_with_dft_squared(data_type)
            title += ' DFT_squared'
        else:
            convolved_data = self.convolve_with_itself(data_type)

        if log_scale:
            convolved_data = np.log(convolved_data + 1)  # Apply logarithmic scale

        plt.figure(figsize=(10, 10))
        plt.imshow(convolved_data, interpolation='nearest')
        plt.colorbar()
        plt.title(title)
        plt.axis('off')  # Hide the axis
        path_fig = os.getcwd()
        plt.savefig(os.path.join(path_fig, f'{title.replace(" ", "_")}.png'))
        print("end plot_heatmap")
        plt.show()


    def compute_fourier_transform(self, data):
        fourier_transform = np.fft.fft2(data)
        fourier_transform_shifted = np.fft.fftshift(fourier_transform)
        return fourier_transform_shifted


    def compute_dft_squared(self, fourier_transform):
        return np.abs(fourier_transform) ** 2


    def inverse_fourier_transform(self, fourier_transform):
        fourier_transform_shifted_back = np.fft.ifftshift(fourier_transform)
        inverse_transform = np.fft.ifft2(fourier_transform_shifted_back)
        return np.abs(inverse_transform)


    def plot_fourier_transform(self, plot_dft_squared = True ,data_type='filtered', title='Fourier Transform of Convolved Data'):
        print("begin plot_fourier_transform")
        convolved_data = self.convolve_with_itself(data_type)
        fourier_transform = self.compute_fourier_transform(convolved_data)
        
        if plot_dft_squared:
            data_to_plot = self.compute_dft_squared(fourier_transform)
            data_to_plot = np.log(data_to_plot + 1)
            title += 'DFT_squared'
        else:
            data_to_plot = np.log(np.abs(fourier_transform) + 1)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(data_to_plot, cmap='gray')
        plt.colorbar()
        plt.title(title)
        plt.axis('off')  # Hide the axis
        path_fig = os.getcwd()
        plt.savefig(os.path.join(path_fig, f'{title.replace(" ", "_")}.png'))
        print("end plot_fourier_transform")
        plt.show()


    def convolve_with_itself_mean(self, image_analysis_list, data_type='filtered'):
        print("begin convolve_with_itself_mean")
        convolved_matrices = []

        for analysis in image_analysis_list:
            interval_filtered_data = analysis.filter_pixels_within_interval(data_type)
            convolved_data = fftconvolve(interval_filtered_data, interval_filtered_data, mode='same')

            # Normalize the convolved data to be between 0 and 1
            convolved_data_min = np.min(convolved_data)
            convolved_data_max = np.max(convolved_data)
            convolved_data_normalized = (convolved_data - convolved_data_min) / (convolved_data_max - convolved_data_min)

            convolved_matrices.append(convolved_data_normalized)

        # Compute the mean of the convolved matrices
        mean_convolved_data = np.mean(convolved_matrices, axis=0)

        # Normalize the mean convolved data to be between 0 and 1
        mean_convolved_data_min = np.min(mean_convolved_data)
        mean_convolved_data_max = np.max(mean_convolved_data)
        mean_convolved_data_normalized = (mean_convolved_data - mean_convolved_data_min) / (mean_convolved_data_max - mean_convolved_data_min)

        print("end convolve_with_itself_mean")
        return mean_convolved_data_normalized
    

    def plot_heatmap_mean(self, image_analysis_list, data_type='filtered', title='Heatmap of Mean Convolved Data', log_scale=False):
        print("begin plot_heatmap_mean")
        mean_convolved_data = self.convolve_with_itself_mean(image_analysis_list, data_type)

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

    
    def plot_fourier_transform_mean(self, image_analysis_list, plot_dft_squared=True, data_type='filtered', title='Fourier Transform of Mean Convolved Data'):
        print("begin plot_fourier_transform_mean")
        mean_convolved_data = self.convolve_with_itself_mean(image_analysis_list, data_type)
        fourier_transform = self.compute_fourier_transform(mean_convolved_data)
        
        if plot_dft_squared:
            data_to_plot = self.compute_dft_squared(fourier_transform)
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