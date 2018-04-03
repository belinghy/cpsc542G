classdef FluidQuantity
  properties
    w
    h
    ox
    oy
    hx
    val
  end
  methods
    % Constructor
    function obj = FluidQuantity(w, h, ox, oy, hx)
      %  h / w  - height / width
      obj.w = w;
      obj.h = h;
      % ox / oy - (0.5, 0.5) for centred quantities
      %   (0.0, 0.5) or (0.5, 0.0) for boundary quantities
      obj.ox = ox;
      obj.oy = oy;
      % hx - height and width of each cell
      obj.hx = hx;
      % src and dst are buffers for speed-up
      obj.val = zeros(w, h);
    end
  end
end