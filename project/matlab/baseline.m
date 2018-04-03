% This work follows parts of Bridson's notes
% https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf

% General stuff for working in 2D
% reshape(A,M,M) is inverse of A(:)

global M; M = 128;  % square grid (MxM)
global hx; hx = 1/M;
global density; density = 0.1;
timestep = 0.005; % How to choose this?
T_MAX = 8;
N_STEPS = T_MAX / timestep;

% fluid quantities in MAC Grid
den_mat = zeros(M,M);
x_vel_mat = zeros(M,M+1); 
y_vel_mat = zeros(M+1,M);

for cur_step=1:N_STEPS
  % set density, x_vel, y_vel in the given rectangle to 1, 0, and 3
  % Problem setup, injection of velocity 
  den_mat = MACSetBlockValue(den_mat, 0.45, 0.2, 0.1, 0.01, 1.0);
  x_vel_mat = MACSetBlockValue(x_vel_mat, 0.45, 0.2, 0.1, 0.01, 0.0);
  y_vel_mat = MACSetBlockValue(y_vel_mat, 0.45, 0.2, 0.1, 0.01, 3.0);
  % Take one step by solving Navier-Stokes
  [den_mat, x_vel_mat, y_vel_mat] = step(den_mat, x_vel_mat, ...
    y_vel_mat, timestep);
  writeImage(den_mat, cur_step)
end

% ----------------------------- %
%        Solver functions       %
% ----------------------------- %
function [dn, xvn, yvn] = step(den_mat, x_vel, y_vel, timestep)
  % Eq. (4.11)
  global M; global density;
  neg_divergence = M*(x_vel(:,2:end)-x_vel(:,1:end-1) + ...
    y_vel(2:end,:)-y_vel(1:end-1,:));
  
  % Calculate pressure
  pressure_matrix = project(neg_divergence, timestep, 600);
  % Make incompressible fluid
  [x_vel, y_vel] = applyPressure(x_vel, y_vel, pressure_matrix, timestep);
  % Calculate next time step
  dn = advect(den_mat, x_vel, y_vel, timestep);
  xvn = advect(x_vel, x_vel, y_vel, timestep);
  yvn = advect(y_vel, x_vel, y_vel, timestep);
end

function p = project(neg_div, timestep, max_iter)
  % Eq. (4.19)
  % Calculate pressure from divergence
  global density; global hx; global M;
  
  scale = timestep / (density*hx*hx);
  A = gallery('poisson', M);
  % neg_div has size (M,M), convert to column
  % pcg is good for poisson
  % It's also general sparse + dense within band, so GEPP?
  X = pcg(A, neg_div(:), 1e-5, max_iter);
  p = reshape(X, M, M) / scale;
end

function [x_vel, y_vel] = applyPressure(x_vel, y_vel, pressure, timestep)
  global hx; global density;
  % Eq. (4.4) and (4.5)
  scale = timestep / (density*hx);
  
  % P_i+1j - P_ij : (M,M-1)
  x_vel(:,2:end-1) = x_vel(:,2:end-1) - ...
    scale*(pressure(:,2:end) - pressure(:,1:end-1));
  
  % P_ij+1 - P_ij : (M-1,M)
  y_vel(2:end-1,:) = y_vel(2:end-1,:) - ...
  scale*(pressure(2:end,:) - pressure(1:end-1,:));
end

function fnew = advect(fquan, x_vel, y_vel, timestep)
  global M;
  % This is semi-lagrangian method
  % Chapter 3 Eq. (3.6) to (3.9)
  width = size(fquan,2); height = size(fquan,1);
  ox = (abs(width-M)-1)/2;
  oy = (abs(height-M)-1)/2;
  
  fnew = zeros(height, width);
  for iy=1:height
    for ix=1:width
      % 1st step is Euler
      x = ix + ox; y = iy + oy;
      xv = M*lerp(x_vel, x, y); yv = M*lerp(y_vel, x, y);
      x = x - xv*timestep; y = y - yv*timestep;
      % 2nd step of Semi-Lagrangian
      fnew(iy,ix) = lerp(fquan,x,y);
    end
  end  
end

% ----------------------------- %
%        Helper functions       %
% ----------------------------- %
function r = lerp(fquan, x, y)
  global M;
  width = size(fquan,2); height = size(fquan,1);
  ox = (abs(width-M)-1)/2;
  oy = (abs(height-M)-1)/2;
  % check for grid bounds
  x = min(max(x-ox, 1.0), width-0.001);
  y = min(max(y-oy, 1.0), height-0.001);
  % index and fractional
  ix = fix(x); iy = fix(y);
  fx = x - ix; fy = y - iy;
  x00 = fquan(iy, ix); x10 = fquan(iy, ix+1);
  x01 = fquan(iy+1, ix); x11 = fquan(iy+1, ix+1);
  
  % Plain old linear interpolation
  linterp = @(a,b,x) a*(1.0-x) + b*x;
  r = linterp(linterp(x00,x10, fx), linterp(x01,x11,fx), fy);
end

function fquan = MACSetBlockValue(fquan, x, y, w, h, val)
  global M;
  %fquan is a 2D matrix, find offsets
  % -1 because if size(fquan)=(M,M), then the value is centered
  width = size(fquan,2); height = size(fquan,1);
  ox = (abs(width-M)-1)/2;
  oy = (abs(height-M)-1)/2;
  % x, y, w, h are given in fractions, need to find index
  ix0 = fix(x*width-ox); iy0 = fix(y*height-oy);
  ix1 = fix((x+w)*width-ox); iy1 = fix((y+h)*height-oy);
  
  % Check array bounds
  for iy=max(iy0,1):min(iy1,height)
    for ix=max(ix0,1):min(ix1,width)
      if abs(fquan(iy,ix)) < abs(val)
        fquan(iy,ix) = val;
      end
    end
  end
end

function writeImage(den_mat, index)
  N = size(den_mat,1);
  rgb = zeros(N, N, 3);
  shade = max(min(fix((1 - den_mat)*255), 255), 0);
  rgb(:,:,1) = shade; rgb(:,:,2) = shade; rgb(:,:,3) = shade; 
  
  alpha = ones(N,N)*255;
  imwrite(rgb, sprintf('frame%05d.png', index), ...
    'png', 'Alpha', alpha);
end
