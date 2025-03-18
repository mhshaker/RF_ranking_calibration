import numpy as np
import time
import multiprocessing

def memory_bandwidth_test(size_in_gb):
    """Test memory read/write speed by copying large arrays."""
    
    size = int(size_in_gb * (1024**3) / 8)  # Convert GB to number of float64 elements
    print(f"\nTesting memory bandwidth with {size_in_gb} GB of data...")

    # Create large arrays
    A = np.random.rand(size)
    B = np.zeros_like(A)

    # Measure write speed (copying data)
    start_write = time.time()
    B[:] = A
    end_write = time.time()
    write_speed = size_in_gb / (end_write - start_write)

    # Measure read speed (accessing data)
    start_read = time.time()
    np.sum(B)
    end_read = time.time()
    read_speed = size_in_gb / (end_read - start_read)

    print(f"Write Speed: {write_speed:.2f} GB/s")
    print(f"Read Speed: {read_speed:.2f} GB/s")

if __name__ == "__main__":
    total_cores = multiprocessing.cpu_count()
    
    # Test with different memory loads
    for gb in [1, 5, 10, 20, 50]:  # Adjust based on your system's RAM
        memory_bandwidth_test(gb)
