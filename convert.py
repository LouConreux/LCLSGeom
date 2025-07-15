import sys
import os
import argparse
from LCLSGeom.converter import PsanaToCrystFEL, PsanaToPyFAI, CrystFELToPyFAI, PyFAIToCrystFEL, PyFAIToPsana, CrystFELToPsana

def main():
    parser = argparse.ArgumentParser(description="Convert geometry files between different formats.")
    parser.add_argument("--in_file", type=str, required=True, help="Path to the input geometry file.")
    parser.add_argument("--format", type=str, choices=["psana", "crystfel", "pyfai"], required=True,
                        help="Output format: psana, crystfel, or pyfai.")
    parser.add_argument("--det_type", type=str, default=None, help="Detector type.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.in_file):
        print(f"Error: Input file '{args.in_file}' does not exist.")
        sys.exit(1)
    
    try:

        in_format = os.path.splitext(args.input_file)[1].lower()

        if args.format == "pyfai":
            print("Cannot convert to PyFAI format from the command-line.")
            print("Use it either in a Jupyter notebook or in a Python script that way:")
            if in_format == ".geom":
                print("converter = PyFAIToCrystFEL(geom_file, det_type, psana_file (0-end.data))")
                print("detector_pyfai = converter.detector")
                print("detector_pyfai is then a PyFAI detector object that can be used for any PyFAI analysis.")
            elif in_format == ".data":
                print("converter = PsanaToPyFAI(data_file, det_type)")
                print("detector_pyfai = converter.detector")
                print("detector_pyfai is then a PyFAI detector object that can be used for any PyFAI analysis.")
            sys.exit(1)

        if args.format == "psana":
            if in_format == ".poni":
                ### Get Psana file in templates folder
                #PyFAIToPsana(args.in_file, psana_file, args.output_file)
                pass
            elif in_format == ".geom":
                ### Get Psana file in templates folder
                #CrystFELToPsana(args.in_file, psana_file, args.output_file)
                pass
            elif in_format == ".data":
                print("Already in psana format !")
                sys.exit(1)
        
        if args.format == "crystfel":
            if in_format == ".poni":
                ### Get Psana file in templates folder
                #PyFAIToCrystFEL(args.in_file, psana_file, args.output_file)
                pass
            elif in_format == ".geom":
                print("Already in CrystFEL format !")
                sys.exit(1)
            elif in_format == ".data":
                out_file = os.path.splitext(args.input_file)[0] + ".geom"
                PsanaToCrystFEL(args.in_file, out_file)

        print(f"Conversion successful: {args.input_file} -> {args.output_file} in {args.format} format.")

    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)