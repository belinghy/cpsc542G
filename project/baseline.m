% This work follows parts of Bridson's notes
% https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf

M = 128;  % square grid (MxM)
hx = 1/M;
density = 0.1;
timestep = 0.005; % How to choose this?
T_MAX = 8;
N_STEPS = T_MAX / timestep;
image = zeros(M, M, 3); % RGA

% fluid quantities in MAC Grid
den_mat = zeros(M,M);
press_mat = zeros(M,M);
x_vel_mat = zeros(M,M+1); 
y_vel_mat = zeros(M+1,M);

for cur_step=1:N_STEPS
  % set density, x_vel, y_vel in the given rectangle to 1, 0, and 3
  % source at roughly centre
  den_mat = MACSetBlockValue(den_mat, 0.45, 0.2, 0.1, 0.01, 1.0);
  x_vel_mat = MACSetBlockValue(x_vel_mat, 0.45, 0.2, 0.1, 0.01, 0.0);
  y_vel_mat = MACSetBlockValue(y_vel_mat, 0.45, 0.2, 0.1, 0.01, 3.0);
  
  % writeImage(image, cur_step)
end

% Problem setup functions

% Solver functions
function [dm, xm, ym] = step(x_vel, y_vel, timestep)
  % Eq. (4.11)
  neg_divergence = M*(x_vel(:,2:end)-x_vel(:,1:end-1) + ...
    y_vel(2:end,:)-y_vel(1:end-1,:));
  
  project(timestep, 600)
  apply_pressue()
  advect()
end

function project(timestep, max_iter)
  % Eq. (4.19)
  scale = timestep / (density*hx*hx);
end

% Helper functions   
function fquan = MACSetBlockValue(fquan, x, y, w, h, val)
  %fquan is a 2D matrix, find offsets
  % -1 because if size(fquan)=(M,M), then the value is centered
  ox = (abs(size(fquan,1)-M)-1)/2;
  oy = (abs(size(fquan,2)-M)-1)/2;
  % x, y, w, h are given in fractions, need to find index
  ix0 = fix(x*M-ox); iy0 = fix(y*M-oy);
  ix1 = fix((x+w)*M-ox); iy1 = fix((y+h)*M-oy);
  
  % Check array bounds
  for iy=max(iy0,1):min(iy1,M)
    for ix=max(ix0,1):min(ix1,M)
      if abs(fquan(iy,ix)) < abs(v)
        fquan(iy,ix) = v;
      end
    end
  end
end

function writeImage(rgb, index)
  alpha = ones(size(rgb,1),size(rgb,2));
  imwrite(rgb, sprintf('frame%05d.png', index), ...
    'png', 'Alpha', alpha);
end
