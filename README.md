# sxdm_nm
Scanning X-ray Diffraction Mapping analysis framework for NanoMAX beamline

The general workflow of SXDM data processing includes the following steps:
- define experimental setup:
    [+] detector
    [+] goniometer angles
    [+] direct beam position
    [+] scan parameters: scan step size
    [+]
- data correction: 
    [+] binning
    [+] cropping
    [+] bad pixel masking
    [+] flat-field correction
    [+] incoming intensity normalization
- processing:
    [+] q-coordinates calculation
    [+] data readout
    [+] center of mass calculation in q-space
- evaluation:
    [+] conversion from q-space to lattice constants
    [+] calculation of strain and tilts of crystalline structure
    [+] plot maps
 