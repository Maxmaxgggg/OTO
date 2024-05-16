function noisy_output = sin_wn(x, sigma, nu)
white_noise = sigma * randn(size(x)) + nu;
noisy_output = sin(x) + white_noise;
end