#!/usr/bin/env python3
import numpy as np
from scipy import fft, constants
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from argparse import ArgumentParser


class Signal:
    def __init__(self, main_file, calibration_file):
        self.name = main_file.stem
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
        self.tick_number = len(self.distance_ticks)
        self.distance_rate = self.distance_ticks[1] - self.distance_ticks[0]
        self.PDP = self.calibrated_PDP()

    def calibrated_PDP(self):
        CIR =  (
            (self.raw_data[:,1] + 1j * self.raw_data[:,2])     
            /
            (self.calibration_data[:,1] + 1j * self.calibration_data[:,2]) 
        )
        PDP = fft.ifft(fft.fftshift(CIR))
        PDP = 10 * np.log10(fft.ifftshift(np.abs(PDP)))
        return PDP

    def theoretical_PDP(self, expected_distance):
        PDP = fft.ifft(
            np.exp(-1j* 2 * np.pi * 
                fft.fftshift(self.frequency_ticks) * 
                expected_distance/constants.c
            )
        )
        PDP = 10 * np.log10(fft.ifftshift(np.abs(PDP)))
        return PDP

    def sinc_interpolate(self, PDP, min_tick = 0., max_tick = 3., res = 0.01): 
        interpolated_distance_ticks = np.arange(min_tick, max_tick + res, res)
        interpolated_PDP = np.zeros_like(interpolated_distance_ticks)
        for i in range(len(interpolated_PDP)):
            interpolated_PDP[i] = np.sum([PDP[n] * 
                np.sinc((interpolated_distance_ticks[i] - 
                    (n - self.tick_number //2 ) * self.distance_rate)
                    /self.distance_rate) 
                for n in range(self.tick_number)])
        return interpolated_distance_ticks, interpolated_PDP
        
    def visualize(self, 
            output_file = 'out', 
            expected_distance = 0.,
            interpolation = True,
            interpolation_res = 0.01, 
            min_tick = 0., 
            max_tick = 3.):

        distance_indices = np.where(np.logical_and(min_tick <= self.distance_ticks, self.distance_ticks <= max_tick))
        ticks = self.distance_ticks[distance_indices]
        PDP = self.PDP[distance_indices]
        max_PDP = np.max(PDP)

        if interpolation:
            int_ticks, int_PDP = self.sinc_interpolate(
                    PDP = self.PDP,
                    min_tick = min_tick, 
                    max_tick = max_tick, 
                    res = interpolation_res)
            max_PDP = np.max(int_PDP)
            int_PDP -= max_PDP
        PDP -= max_PDP

        if expected_distance > 0:
            theo_ticks, theo_PDP = self.sinc_interpolate(
                    PDP = self.theoretical_PDP(
                        expected_distance=expected_distance),
                    min_tick = min_tick, 
                    max_tick = max_tick, 
                    res = interpolation_res)

        _, ax = plt.subplots()
        ax.set_title(r'Power Delay Profile vs Delay $\times$ Speed of light')
        ax.set_xlim(left=min_tick, right=max_tick)
        # ax.set_ylim(bottom = -30., top=0.)
        ax.set_xlabel(r"Delay $\times$ Speed of light [m]")
        ax.set_ylabel("Normalized power delay profile [dB]")
        marker = 's'
        markersize = 5.
        linestyle = '-'
        if expected_distance <= 0.:
            ax.plot(ticks,
                    PDP,
                    color = 'red',
                    linestyle = 'None',
                    marker = marker,
                    markersize = markersize,
            )
            if interpolation:
                ax.plot(int_ticks,
                        int_PDP,
                        color = 'red',
                        linestyle = linestyle,
                        )
        else:
            ax.plot(ticks,
                    PDP,
                    color = 'red',
                    linestyle = 'None',
                    marker = marker,
                    markersize = markersize,
                    label = "Discrete result"
            )
            if interpolation:
                ax.plot(int_ticks,
                        int_PDP,
                        color = 'red',
                        linestyle = linestyle,
                        label = "Sinc interpolation result"
                    )
            ax.plot(theo_ticks,
                    theo_PDP,
                    color = 'blue',
                    linestyle = linestyle,
                    label = "Theoretical calculation"
            )
            ax.legend()
        plt.savefig(output_file + self.name + '.png', dpi=300)
        plt.close()
        
def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_file", 
            type=str, 
            action="store", 
            metavar='csv', 
            help = "Measurement signal from VNA (Ventor Network Analyzer)",
            default=''
            )
    parser.add_argument("-f", "--input_folder", 
            type=str, 
            action="store", 
            metavar='folder', 
            help = "Folder containing measurement signal from VNA (Ventor Network Analyzer)",
            default=''
            )
    parser.add_argument("-c", "--calibration_file", 
            type=str, 
            action="store", 
            metavar='csv', 
            help = "Calibration signal from VNA (Ventor Network Analyzer)", 
            required=True)
    parser.add_argument("-o", "--output_file", 
            type=str, 
            action="store", 
            metavar='csv', 
            help = "Desired output file name", 
            default='out')
    parser.add_argument("-e", "--expected_distance", 
            type=float, 
            action="store", 
            metavar="distance in meter", 
            help="expected distance between Transmitter and Receiver", 
            default=0.)
    parser.add_argument("-no_int", "--no_interpolation", 
            action="store_true", 
            default=False, 
            help="Not using sinc interpolation?")
    parser.add_argument("-res", "--interpolation_res", 
            type=float, action="store", 
            metavar="interpolation resolution", 
            help="interpolation resolution for sinc interpolation", 
            default=0.01)
    parser.add_argument("-min", "--minimum_range", 
            type=float, 
            action="store", 
            metavar="distance in meter", 
            help="left side of the delay graph", 
            default=0.)
    parser.add_argument("-max", "--maximum_range", 
            type=float, 
            action="store", 
            metavar="distance in meter", 
            help="right side of the delay graph", 
            default=4.)

    args = vars(parser.parse_args())
    if len(args["input_file"]) > 0:
        input_files = [Path(args["input_file"])]
    else:
        if len(args["input_folder"]) == 0:
            raise ValueError("Need to provide either file or folder input")
        else:
            input_files = list(Path(args["input_folder"]).glob('*.csv'))

    for input_file in input_files:
        signal = Signal(input_file, args["calibration_file"])
        signal.visualize(
                output_file = args["output_file"],
                expected_distance = args["expected_distance"],
                interpolation = not args["no_interpolation"],
                interpolation_res = args["interpolation_res"],
                min_tick = args["minimum_range"],
                max_tick = args["maximum_range"]
                )

    print(f"Input signal file: {args['input_file']}.")
    print(f"Input calibration file: {args['calibration_file']}.")
    if args["expected_distance"] > 0.:
        print(f"Using theoretical calculation with expected distance of {args['expected_distance']} m.")
    # print(f"Distance resolution after interpolation: {args['interpolation_res']} m.")
    print(f"x-axis spans from {args['minimum_range']} m to {args['maximum_range']} m.")
    print(f"Output file/directory: {args['output_file']}.")

if __name__=='__main__':
    main()
