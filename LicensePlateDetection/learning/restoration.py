"""Optional Gaussian-PSF Wiener restoration; parameters are an assumption."""
import numpy as np


def wiener_restore(gray,kernel_size=5,sigma=1.,balance=.03):
    if gray.ndim!=2 or kernel_size<3 or kernel_size%2==0 or sigma<=0 or balance<=0:
        raise ValueError('Invalid Wiener restoration parameters')
    radius=kernel_size//2
    padded=np.pad(gray.astype(np.float32),radius,mode='reflect')
    axis=np.arange(kernel_size)-radius
    kernel=np.exp(-(axis[:,None]**2+axis[None,:]**2)/(2*sigma*sigma));kernel/=kernel.sum()
    psf=np.zeros_like(padded);psf[:kernel_size,:kernel_size]=kernel
    psf=np.roll(psf,(-radius,-radius),(0,1))
    transfer=np.fft.fft2(psf)
    restored=np.fft.ifft2(np.fft.fft2(padded)*np.conj(transfer)/(np.abs(transfer)**2+balance)).real
    # Preserve the DC level; regularization otherwise darkens a constant image.
    restored*=1+balance
    return np.clip(restored[radius:-radius,radius:-radius],0,255).astype(np.uint8)
