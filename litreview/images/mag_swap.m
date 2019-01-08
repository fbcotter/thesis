x1 = double(imread('~/data/lena256.bmp'));
x2 = double(imread('cameraman.tif'));

y1 = fft2(x1);
y2 = fft2(x2);

% Swap their magnitudes

y1_hat = abs(y2).* exp(1i*angle(y1));
y2_hat = abs(y1) .* angle(y2);

x1_hat = ifft2(y1_hat);
x2_hat = ifft2(y2_hat);

figure; subplot(1,3,1); imshow(uint8(x1));
set(gca, 'Position', [0.02 0.1 0.3 0.8]);
subplot(1,3,2); imshow(uint8(x2));
set(gca, 'Position', [0.35 0.1 0.3 0.8]);
subplot(1,3,3); imagesc(real(x1_hat)); axis image; axis off;
set(gca, 'Position', [0.68 0.1 0.3 0.8]);
set(gcf, 'Position', [477 1410 905 341]);

figure; subplot(1,3,1); imagesc(log(x1)); axis image; axis off;
set(gca, 'Position', [0.02 0.1 0.3 0.8]);
subplot(1,3,2); imagesc(x1.^2); axis image; axis off; colormap gray;
set(gca, 'Position', [0.35 0.1 0.3 0.8]);
subplot(1,3,3); imagesc(x1.^4-10*x1.^3 + 5); axis image; axis off; colormap gray;
set(gca, 'Position', [0.68 0.1 0.3 0.8]);
set(gcf, 'Position', [477 1410 905 341]);
