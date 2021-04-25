%{
Copyright 2018 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS-IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
%}

% Plots the band splitting filters and tests the coefficients by checking
% if a sawtooth signal with the fundamental frequency set to filter cutoff
% frequency is properly attenuated by the low-pass filter.

clearvars
close all
clc

SAMPLE_RATE = 48000;
CUTOFF = 500;
SINE_TONE_FREQUENCY = CUTOFF;

% Sawtooth signal.
T = 1 / SAMPLE_RATE;          % Sample period
L = 2000;                     % Length of signal
t = (0:L-1)* T;               % Time vector
x = sawtooth(2 * pi * SINE_TONE_FREQUENCY * t);

L = length(x);
NFFT = 2 ^ nextpow2(L);
X = fft(x, NFFT) / L;
f = SAMPLE_RATE / 2 * linspace(0, 1, NFFT / 2 + 1);

% Plot single-sided amplitude spectrum.
figure(1)
semilogx(f,2 * abs(X(1:NFFT / 2 + 1)))
axis([10 20000 0 0.6])
title('Single-Sided amplitude spectrum of the sawtooth signal')
xlabel('Frequency (Hz)')
ylabel('|X(f)|')
grid on

% Filter design.
[bLp, bHp, a] = bandsplittingfilterscoeffs(CUTOFF, SAMPLE_RATE);
[hlp, w] = freqz(bLp, a);
hHp = freqz(bHp, a);

% Plot single-sided amplitude spectrum of the band splitting filters.
figure(2)
subplot(2, 1, 1);
semilogx(w / pi * SAMPLE_RATE / 2, 20 * log10(abs(hlp)))

hold on

semilogx(w / pi * SAMPLE_RATE / 2, 20 * log10(abs(hHp)),'r')
title('Single-Sided amplitude spectrum of the filters')
xlabel('Frequency (Hz)')
ylabel('20log10(|H(f)|)')
axis([10 20000 -20 0])
grid on

subplot(2, 1, 2);
semilogx(w / pi * SAMPLE_RATE / 2, angle(hlp))

hold on

semilogx(w / pi * SAMPLE_RATE / 2, angle(hHp))
title('Single-Sided Phase Spectrum of the filters')
xlabel('Frequency (Hz)')
ylabel('phase of H(f)')
axis([10 20000 -pi pi])
grid on

hold off

% filter x with the low-pass filter.
y = filter(bLp, a, x);

L = length(y);
NFFT = 2 ^ nextpow2(L);
Y = fft(y, NFFT) / L;
f = SAMPLE_RATE / 2 * linspace(0, 1, NFFT / 2 + 1);

% Plot filtered spectrum.
figure(3)
semilogx(f, 2 * abs(Y(1:NFFT / 2 + 1)))
axis([10 20000 0 0.6])
title('Single-Sided Amplitude Spectrum of the filtered sawtooth signal.')
xlabel('Frequency (Hz)')
ylabel('|Y(f)|')
grid on
