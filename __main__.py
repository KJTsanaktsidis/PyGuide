"""
PyGuide2, A python program for doing things with X-Ray Waveguides

"""
import waveguides
import argparse
import actions
import numpy as np

if __name__ == '__main__':

    #Parse them arguments!
    parser = argparse.ArgumentParser(description="A program for solving problems related to waveguide guided modes")

    #options
    parser.add_argument('-v', '--verbose', action='store_true',
        help='Display more information about the execution of the program')
    parser.add_argument('-wg', '--waveguide', action='store', default='losswg',
        help='The waveguide to perform calculations on (eg "losswg" or "perfectwg")')
    parser.add_argument('-wgf', '--waveguidefile', action='store',
        help='An XML file describing a waveguide to perform calculations on (takes precedence over -wg)')
    parser.add_argument('-o', '--output', action='store', required=True,
        help='File in which to store the output of this program. If multiple files are generated, this name will be' +
                ' taken as a directory name to create')

    #what-to-do
    subparsers = parser.add_subparsers(dest='subparser')

    #mode-solver
    ms_parser = subparsers.add_parser('modesolver')
    ms_action = ms_parser.add_mutually_exclusive_group(required=True)

    ms_action.add_argument('-kx', '--wavevectors', action='store_true',
        help='Find the wavevectors for the guided modes in this waveguide, in both the real and lossless case')
    ms_action.add_argument('-mp', '--modeplots', action='store_true',
        help='Produce plots of guided modes in this waveguide')
    ms_action.add_argument('-mlp', '--modelossplots', action='store_true',
        help='Produce a plot of guided modes, showing their loss')
    ms_action.add_argument('-mpp', '--modephaseplots', action='store_true',
        help='Produce a plot of guided modes, showing their phase gradient')
    ms_action.add_argument('-ac', '--attncoeffs', action='store_true',
        help='Print the effective linear attenuation coefficients of this waveguides modes')
    ms_action.add_argument('-wf', '--wavefunctions', action='store_true',
        help='Produce a plot of the real and imaginary part of the wavefunctions of guided modes')

    ms_parser.add_argument('-m', '--modes', nargs='+', type=int,
        help='The guided modes upon which to operate (check --wavevectors to get the number of guided modes)')
    ms_parser.add_argument('-d', '--distances', nargs='+', type=float,
        help='The distances at which to produce plots (relevant to --modelossplots)')
    ms_parser.add_argument('-l', '--wavelength', action='store', default=1.54e-10, type=float,
        help='The wavelength of light that is illuminating this waveguide (default copper Ka1)')

    #get args
    args = parser.parse_args()

    #now decide what to do

    #Get the correct waveguide
    if args.waveguidefile is not None:
        waveguide = waveguides.PlanarWaveguideInterp(args.waveguidefile)
    else:
        waveguide = waveguides.PlanarWaveguideInterp.get_waveguide(args.waveguide)

    if args.subparser == 'modesolver':
        #What guided mode numbers?
        modes = args.modes
        #Triggered the modesolver. What action exactly?
        if args.wavevectors: #Print the kx values of modes
            kxs, angs = actions.modesolver_find_kx(waveguide, args.wavelength, args.verbose)
            with open(args.output, 'wb') as f:
                actions.modesolver_output_kx(kxs, angs, f)
        elif args.modeplots: #Plot of guided mode (s)
            actions.modesolver_plot_modes(args.modes, waveguide, args.wavelength, args.output, verbose=args.verbose)
        elif args.modelossplots: #Loss plots(s)
            actions.modesolver_plot_loss(args.modes, waveguide, args.wavelength, args.distances, args.output,
                verbose=args.verbose)
        elif args.modephaseplots: #Phase plot(s)
            actions.modesolver_plot_phase(args.modes, waveguide, args.wavelength, args.output, verbose=args.verbose)
        elif args.attncoeffs:
            kxs, angs = actions.modesolver_find_kx(waveguide, args.wavelength, args.verbose)
            k = 2*np.pi/args.wavelength
            with open(args.output, 'wb') as f:
                actions.modesolver_output_attncoeffs(k, kxs, angs, f)
        elif args.wavefunctions:
            actions.modesolver_plot_wavefunctions(args.modes, waveguide, args.wavelength, args.output, verbose=args.verbose)