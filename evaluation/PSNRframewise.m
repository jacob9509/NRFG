function PSNR = PSNRframewise( I, I_denoised )

for i = 1:size(I,2)
    PSNR_framewise(i)=20*log10(1 * sqrt(numel(I_denoised(:,i))) / norm(I_denoised(:,i)-I(:,i)));
end
PSNR = mean(PSNR_framewise);
end

