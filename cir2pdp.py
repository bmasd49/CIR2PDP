#!/usr/bin/env python3
import numpy as np
from scipy import fft, constants, signal, interpolate
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser


class Signal:
    def __init__(self, main_file, calibration_file):
        self.raw_data = \
                pd.read_csv(main_file, skiprows=2, usecols=range(3)).to_numpy()
        self.calibration_data = \
                pd.read_csv(calibration_file, skiprows=2, usecols=range(3))\
                .to_numpy()
        self.frequency_ticks = self.raw_data[:,0]
        bandwidth = self.frequency_ticks[-1] - self.frequency_ticks[0]
        df = self.frequency_ticks[1] - self.frequency_ticks[0]
        time_ticks = np.arange(-1/(2*df), 1/(2*df)+1/bandwidth, 
                1/(bandwidth))
        self.distance_ticks = constants.c * time_ticks

    def normalized_power_delay_profile(self):
        complex_signal_in_freq_domain =  (
            (self.raw_data[:,1] + 1j * self.raw_data[:,2])     
            /
            (self.calibration_data[:,1] + 1j * self.calibration_data[:,2]) 
            )
        complex_signal_in_time_domain = \
                fft.ifft(fft.fftshift(complex_signal_in_freq_domain))
        signal_in_time_domain = np.abs(complex_signal_in_time_domain)
        signal_in_time_domain = fft.ifftshift(signal_in_time_domain)
        signal_in_time_domain = 10 * np.log10(signal_in_time_domain)
        signal_in_time_domain -= np.max(signal_in_time_domain)
        return signal_in_time_domain

    def theoretical_power_delay_profile(self, expected_distance):
        signal_in_time_domain = \
                fft.ifft(np.exp(-1j* 2 * np.pi * fft.fftshift(self.frequency_ticks) * expected_distance/constants.c))
        signal_in_time_domain = np.abs(signal_in_time_domain)
        signal_in_time_domain = fft.ifftshift(signal_in_time_domain)
        signal_in_time_domain = 10 * np.log10(signal_in_time_domain)
        return signal_in_time_domain

    def visualize(self, output_file, expected_distance = 0.):
        _, ax = plt.subplots()
        ax.set_title(r'Power Delay Profile vs Delay $\times$ Speed of light')
        ax.set_xlim(left=0, right=3.5)
        ax.set_xlabel(r"Delay $\times$ Speed of light [m]")
        ax.set_ylabel("Normalized power delay profile [dB]")
        marker = 's'
        markersize = 5.
        if expected_distance <= 0.:
            ax.plot(self.distance_ticks, 
                    self.normalized_power_delay_profile(),
                    color = 'red',
                    linestyle = 'None',
                    marker = marker,
                    markersize = markersize,
                    )
        else:
            ax.plot(self.distance_ticks, 
                    self.normalized_power_delay_profile(),
                    color = 'red',
                    linestyle = 'None',
                    marker = marker,
                    markersize = markersize,
                    label = "Experimental"
                    )
            ax.plot(self.distance_ticks, 
                    self.theoretical_power_delay_profile(expected_distance), 
                    color = 'blue',
                    linestyle = 'None',
                    marker = marker,
                    markersize = markersize,
                    label = "Theoretical"
                    )
            ax.legend()
        plt.savefig(output_file, dpi=300)
        plt.close()
        
def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, action="store", metavar='csv', help = "Measurement signal from VNA (Ventor Network Analyzer)", required=True)
    parser.add_argument("-o", "--output_file", type=str, action="store", metavar='csv', help = "Desired output file name", default='out.png')
    parser.add_argument("-c", "--calibration_file", type=str, action="store", metavar='csv', help = "Calibration signal from VNA (Ventor Network Analyzer)", required=True)
    parser.add_argument("-e", "--expected_distance", type=float, action="store", metavar="distance in meter", help="expected distance between Transmitter and Receiver", default=0.)
    args = vars(parser.parse_args())
    signal = Signal(args["input_file"], args["calibration_file"])
    signal.visualize(output_file= args["output_file"], expected_distance = args["expected_distance"])

if __name__=='__main__':
    main()
