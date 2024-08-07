# %% S0. SETUP env
import MRzeroCore as mr0
import pypulseq as pp
import numpy as np
import torch
from matplotlib import pyplot as plt
# makes the ex folder your working directory
import os 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.chdir(os.path.abspath(os.path.dirname(__file__)))

experiment_id = 'T_GRE_zigzag_nufft'

# %% S1. SETUP sys

# choose the scanner limits
system = pp.Opts(
    max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s',
    rf_ringdown_time=20e-6, rf_dead_time=100e-6,
    adc_dead_time=20e-6
)


# %% S2. DEFINE the sequence
seq = pp.Sequence(system) 

# Define FOV and resolution
fov = 1000e-3
slice_thickness = 8e-3
sz = (64, 64)   # spin system size / resolution
Nread = 64     # frequency encoding steps/samples
Nphase = 64    # phase encoding steps/samples - number of spokes

# Define rf events
rf1, _, _ = pp.make_sinc_pulse(
    flip_angle=180 * np.pi / 180, duration=1e-3,
    slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4,
    system=system, return_gz=True
)
# rf1 = pp.make_block_pulse(flip_angle=90 * np.pi / 180, duration=1e-3, system=system)
rf0, _, _ = pp.make_sinc_pulse(
    flip_angle=90 * np.pi / 180, duration=1e-3,
    slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4,
    system=system, return_gz=True
)

# Define other gradients and ADC events


rf_phase = 180
rf_inc = 180



# ======
# CONSTRUCT SEQUENCE
# ======
sdel = 1e-0
t=1e-3

seq.add_block(rf0)
seq.add_block(pp.make_delay(3e-3))
seq.add_block(rf1)
seq.add_block(pp.make_delay(9e-3))

gx = pp.make_trapezoid(channel='x', area=Nread, duration=1e-3, system=system)
gx_prewinder = pp.make_trapezoid(channel='x', area= -gx.area, duration=1e-3, system=system)
adc0 = pp.make_adc(num_samples=Nread, duration=1e-3, phase_offset=0 * np.pi / 180, delay=gx.rise_time, system=system)
gx_prewinder = pp.make_trapezoid(channel='x', area= -gx.area/2, duration=1e-3, system=system)
gy_prewinder = pp.make_trapezoid(channel='y', area= -gx.area/2, duration=1e-3, system=system)

seq.add_block(gx_prewinder,gy_prewinder)  
gx = pp.make_trapezoid(channel='x', area=Nread, duration=1e-3, system=system)

r = gx.rise_time
flt = gx.flat_time
fll = gx.fall_time


gxe = pp.make_extended_trapezoid(channel = 'x', amplitudes=np.array([0, gx.amplitude, gx.amplitude,
                                                                    0, -gx.amplitude, - gx.amplitude,
                                                                    0, gx.amplitude, gx.amplitude, 0]), system=system,
                                times=np.array([0, r, r+flt, 
                                                r+flt+fll, 2*r+flt+fll, 2*r+2*flt+fll, 
                                                2*r+2*flt+2*fll, 3*r+2*flt+2*fll, 3*r+3*flt+2*fll,
                                                3*r+3*flt+3*fll
                                                ]))

amp = np.array([])
for ii in range(0,Nphase):
    amp=np.append(amp,[0, gx.amplitude, gx.amplitude,0, -gx.amplitude, -gx.amplitude])
amp=np.append(amp,0)
plt.plot(amp)

text=gx.rise_time+gx.flat_time+gx.fall_time+gx.rise_time+gx.flat_time+gx.fall_time
t = np.array([])
for ii in range(0,Nphase):
    t=np.append(t,ii*text+np.array([0, 
                                    gx.rise_time, 
                                    gx.rise_time+gx.flat_time,
                                    gx.rise_time+gx.flat_time+gx.fall_time,
                                    gx.rise_time+gx.flat_time+gx.fall_time+gx.rise_time,
                                    gx.rise_time+gx.flat_time+gx.fall_time+gx.rise_time+gx.flat_time]))
t=np.append(t,t[-1]+gx.fall_time)
plt.plot(t)

plt.plot(t,amp)

gxe = pp.make_extended_trapezoid(channel = 'x', amplitudes=amp, system=system, times=t)



# %%
a=1
b=1
c =1
def generate_pattern_time(n):
    for i in range(1,n+1):
        amp = []
        amp.append(i*a)
        amp.append(i*a+i*b)
        amp.append(i*a+i*b+i*c)
    return amp

time = generate_pattern_time(n=4)
time.insert(0, 0)
print(time)

# %%
gx.amplitude*=2.5
# gx = pp.make_extended_trapezoid(channel = 'x', amplitudes=np.array([0, gx.amplitude, gx.amplitude,
#                                                                     0, -gx.amplitude, -gx.amplitude,
#                                                                     0, gx.amplitude, gx.amplitude,
#                                                                     0, -gx.amplitude, -gx.amplitude,
#                                                                     0, gx.amplitude, gx.amplitude,
#                                                                     0, -gx.amplitude, -gx.amplitude,
#                                                                     0, gx.amplitude, gx.amplitude,
#                                                                     0, -gx.amplitude, -gx.amplitude,
#                                                                     0, gx.amplitude, gx.amplitude,
#                                                                     0, -gx.amplitude, -gx.amplitude,
#                                                                     0, gx.amplitude, gx.amplitude,
#                                                                     0, -gx.amplitude, -gx.amplitude,
#                                                                     0, gx.amplitude, gx.amplitude,
#                                                                     0, -gx.amplitude, -gx.amplitude,
#                                                                     0, gx.amplitude, gx.amplitude,
#                                                                     0, -gx.amplitude, -gx.amplitude,
#                                                                     0, gx.amplitude, gx.amplitude,
#                                                                     0, -gx.amplitude, -gx.amplitude,
#                                                                     0, gx.amplitude, gx.amplitude,
#                                                                     0, -gx.amplitude, -gx.amplitude,
#                                                                     0])
#                                 , system=system,
#                                 times=np.array([0, t/5, 2*t/5, 
#                                                 #gx.rise_time, gx.rise_time+gx.flat_time, gx.rise_time+gx.flat_time+gx.fall_time,
#                                                 3*t/5, 4*t/5, t,
#                                                 6*t/5, 7*t/5, 8*t/5,
#                                                 9*t/5, 2*t, 11*t/5,
#                                                 12*t/5, 13*t/5, 14*t/5,
#                                                 3*t, 16*t/5, 17*t/5,
#                                                 18*t/5, 19*t/5, 4*t,
#                                                 21*t/5, 22*t/5, 23*t/5,
#                                                 24*t/5,5*t, 26*t/5, 
#                                                 27*t/5,28*t/5, 29*t/5,
#                                                 6*t, 31*t/5, 32*t/5, 
#                                                 33*t/5,34*t/5,35*t/5, 
#                                                 36*t/5, 37*t/5, 38*t/5, 
#                                                 39*t/5, 40*t/5, 41*t/5, 42*t/5,
#                                                 43*t/5, 44*t/5, 
#                                                 45*t/5, 46*t/5, 47*t/5, 48*t/5,
#                                                 49*t/5, 50*t/5, 
#                                                 51*t/5, 52*t/5, 53*t/5, 54*t/5,
#                                                 55*t/5, 56*t/5, 
#                                                 57*t/5, 58*t/5, 59*t/5, 60*t/5
#                                                 ]))

# gy = pp.make_trapezoid(channel='y', area=Nread, duration=1e-3, system=system)
# gy = pp.make_extended_trapezoid(channel = 'y', amplitudes=np.array([0, gy.amplitude, gy.amplitude,0]), system=system,
                                # times=np.array([0,1.87*t,5.62*t,7.5*t]))
adc = pp.make_adc(num_samples=Nread*Nphase, duration=t[-1], phase_offset=0 * np.pi / 180, delay=gx.rise_time, system=system)
gy = pp.make_trapezoid(channel='y', area=Nread, duration=t[-1], system=system)
seq.add_block(adc,gxe,gy)

#pp.add_gradients()
    


# %% S3. CHECK, PLOT and WRITE the sequence  as .seq
# Check whether the timing of the sequence is correct
ok, error_report = seq.check_timing()
if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed. Error listing follows:')
    [print(e) for e in error_report]

# PLOT sequence
sp_adc, t_adc = mr0.util.pulseq_plot(seq, clear=False, figid=(11,12))

# Prepare the sequence output for the scanner
seq.set_definition('FOV', [fov, fov, slice_thickness])
seq.set_definition('Name', 'gre')
seq.write('out/external.seq')
seq.write('out/' + experiment_id + '.seq')


# %% S4: SETUP SPIN SYSTEM/object on which we can run the MR sequence external.seq from above

if 1:
    # (i) load a phantom object from file
    # obj_p = mr0.VoxelGridPhantom.load_mat('../data/phantom2D.mat')
    obj_p = mr0.VoxelGridPhantom.load_mat('../data/numerical_brain_cropped.mat')
    obj_p = obj_p.interpolate(sz[0], sz[1], 1)

# Manipulate loaded data
    obj_p.T2dash[:] = 30e-3
    obj_p.D *= 0
    obj_p.B0 *= 1    # alter the B0 inhomogeneity
    # Store PD for comparison
    PD = obj_p.PD.squeeze()
    B0 = obj_p.B0.squeeze()
else:
    # or (ii) set phantom  manually to a pixel phantom. Coordinate system is [-0.5, 0.5]^3
    obj_p = mr0.CustomVoxelPhantom(
        pos=[[-0.4, -0.4, 0], [-0.4, -0.2, 0], [-0.3, -0.2, 0], [-0.2, -0.2, 0], [-0.1, -0.2, 0]],
        PD=[1.0, 1.0, 0.5, 0.5, 0.5],
        T1=1.0,
        T2=0.1,
        T2dash=0.1,
        D=0.0,
        B0=0,
        voxel_size=0.1,
        voxel_shape="box"
    )
    # Store PD for comparison
    PD = obj_p.generate_PD_map()
    B0 = torch.zeros_like(PD)

#obj_p.plot()
obj_p.size=torch.tensor([fov, fov, slice_thickness]) 
# Convert Phantom into simulation data
obj_p = obj_p.build()

#obj_p.T1[:7]


# %% S5:. SIMULATE  the external.seq file and add acquired signal to ADC plot

# Read in the sequence 
seq0 = mr0.Sequence.import_file("out/external.seq")
 
seq0.plot_kspace_trajectory()
kspace_loc = seq0.get_kspace()
# Simulate the sequence
graph = mr0.compute_graph(seq0, obj_p, 200, 1e-3)
signal = mr0.execute_graph(graph, seq0, obj_p)

# PLOT sequence with signal in the ADC subplot
plt.close(11);plt.close(12)
sp_adc, t_adc = mr0.util.pulseq_plot(seq, clear=False, signal=signal.numpy())


# %% S6: MR IMAGE RECON of signal ::: #####################################
fig = plt.figure()  # fig.clf()
plt.subplot(411)
plt.title('ADC signal')


# # Verify the shape of the signal
# print(f"Signal size: {signal.size()}")

# # Check the expected size
# expected_size = Nphase * Nread
# if signal.size(0) != expected_size:
#     print(f"Warning: Expected signal size {expected_size}, but got {signal.size(0)}")

# # Adjust the reshaping if necessary
# if signal.size(0) == 2 * expected_size:  # If the signal is twice as large as expected
#     signal = signal[:expected_size]  # Trim the signal to the expected size
# elif signal.size(0) != expected_size:
#     raise ValueError(f"Unexpected signal size: {signal.size(0)}")


kspace_adc = torch.reshape((signal), (Nphase, Nread)).clone().t()
plt.plot(torch.real(signal), label='real')
plt.plot(torch.imag(signal), label='imag')

# this adds ticks at the correct position szread
major_ticks = np.arange(0, Nphase * Nread, Nread)
ax = plt.gca()
ax.set_xticks(major_ticks)
ax.grid()

if 0:  # FFT
    # fftshift
    spectrum = torch.fft.fftshift(kspace_adc)
    # FFT
    space = torch.fft.fft2(spectrum)
    # fftshift
    space = torch.fft.fftshift(space)


if 1:  # NUFFT
    import scipy.interpolate
    grid = kspace_loc[:, :2]
    Nx = 64
    Ny = 64

    X, Y = np.meshgrid(np.linspace(0, Nx - 1, Nx) - Nx / 2,
                       np.linspace(0, Ny - 1, Ny) - Ny / 2)
    grid = np.double(grid.numpy())
    grid[np.abs(grid) < 1e-3] = 0

    plt.subplot(347)
    plt.plot(grid[:, 0].ravel(), grid[:, 1].ravel(), 'rx', markersize=3)
    plt.plot(X, Y, 'k.', markersize=2)
    plt.show()

    spectrum_resampled_x = scipy.interpolate.griddata(
        (grid[:, 0].ravel(), grid[:, 1].ravel()),
        np.real(signal.ravel()), (X, Y), method='cubic'
    )
    spectrum_resampled_y = scipy.interpolate.griddata(
        (grid[:, 0].ravel(), grid[:, 1].ravel()),
        np.imag(signal.ravel()), (X, Y), method='cubic'
    )

    kspace_r = spectrum_resampled_x + 1j * spectrum_resampled_y
    kspace_r[np.isnan(kspace_r)] = 0

    # fftshift
    # kspace_r = np.roll(kspace_r,Nx//2,axis=0)
    # kspace_r = np.roll(kspace_r,Ny//2,axis=1)
    kspace_r_shifted = np.fft.fftshift(kspace_r, 0)
    kspace_r_shifted = np.fft.fftshift(kspace_r_shifted, 1)

    space = np.fft.fft2(kspace_r_shifted)
    space = np.fft.fftshift(space, 0)
    space = np.fft.fftshift(space, 1)

space = np.transpose(space)
plt.subplot(345)
plt.title('k-space')
mr0.util.imshow(np.abs(kspace_adc))
plt.subplot(349)
plt.title('k-space_r')
mr0.util.imshow(np.abs(kspace_r))

plt.subplot(346)
plt.title('FFT-magnitude')
mr0.util.imshow(np.abs(space))
plt.colorbar()
plt.subplot(3, 4, 10)
plt.title('FFT-phase')
mr0.util.imshow(np.angle(space), vmin=-np.pi, vmax=np.pi)
plt.colorbar()

# % compare with original phantom obj_p.PD
plt.subplot(348)
plt.title('phantom PD')
mr0.util.imshow(obj_p.recover().PD.squeeze())
plt.subplot(3, 4, 12)
plt.title('phantom B0')
mr0.util.imshow(obj_p.recover().B0.squeeze())