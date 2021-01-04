# RF Zero Shot Learning
This repository is intended store the data caode for the RF ZSL Project

## data

This folder stores the sample data for testing and experimentation.

These file are written as 16-bit signed integers in little endian format.

The data is stored in an interleaved real, imaginary, real, imaginary, etc...

  - rand_test_10M_100m_0000.bin: random IQ data with sample rate of 10MHz, a duration of 100ms and a maximum value of 2047 and a minimum value of -2048
  - rand_test_10M_100m_0001.bin: random IQ data with sample rate of 10MHz, a duration of 100ms and a maximum value of 1023 and a minimum value of -1024
  - rand_test_10M_100m_0002.bin: random IQ data with sample rate of 10MHz, a duration of 100ms and a maximum value of 255 and a minimum value of -256
  - lfm_test_10M_100m_0000.bin: linear frequency modulation IQ data with sample rate of 10MHz, a duration of 100ms, a starting frequcny of -2Mhz, an ending frequency of 2Mhz and a maximum value of 2047 and a minimum value of -2048
  - lfm_test_10M_100m_0001.bin: linear frequency modulation IQ data with sample rate of 10MHz, a duration of 100ms, a starting frequcny of -2Mhz, an ending frequency of 2Mhz and a maximum value of 1023 and a minimum value of -1024
  - lfm_test_10M_100m_0002.bin: linear frequency modulation IQ data with sample rate of 10MHz, a duration of 100ms, a starting frequcny of -2Mhz, an ending frequency of 2Mhz and a maximum value of 255 and a minimum value of -256

## Reading the Data

The data can be read in using Python or C++ or any other library that can open and read binary files.

Python: the data can be read in by any number of packages.  The numpy package provides a simple one line method to bring in the data and store as a 32-bit float.

```
import numpy as np
data = np.fromfile("rf_zsl/data/rand_test_10M_100m_0000.bin", dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)
```

C++: The methodology is still being worked
