import numpy as np
import argparse

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Load and display data from a .npy file')
    parser.add_argument('file_path', type=str, help='Path to the .npy file')
    args = parser.parse_args()

    # Get the file path from the command line arguments
    npy_file_path = args.file_path

    try:
        # Load the .npy file with allow_pickle=True to handle object arrays
        data = np.load(npy_file_path, allow_pickle=True)
        
        # Print the contents of the file
        print("Data loaded from .npy file:")
        print(data)
        
        # Print the shape of the data
        print("Shape of the data:", data.shape)
        
        # Print the first 5 elements if it's a 1D array
        if data.ndim == 1:
            print("First 5 elements:", data[:5])
        elif data.ndim == 2:  # Example for 2D array
            print("First 5 rows:")
            print(data[:5, :])

        # Perform a simple computation (e.g., mean of the data) if possible
        if np.issubdtype(data.dtype, np.number):
            mean_value = np.mean(data)
            print("Mean value of the data:", mean_value)
        else:
            print("Data is not numeric, skipping mean calculation.")
    except Exception as e:
        print(f"Error loading .npy file: {e}")

if __name__ == "__main__":
    main()
